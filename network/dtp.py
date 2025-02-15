import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class DTPLossConfig:
    """Configuration for DTP loss calculation."""
    beta: float = 0.9
    noise_scale: float = 0.1
    K_iterations: int = 1


class DTPLayer(nn.Module):
    """An improved DTP layer with local gradient control."""

    def __init__(self, forward_layer: nn.Module):
        super().__init__()
        self.forward_layer = forward_layer
        self.feedback_layer = self._create_feedback_layer(forward_layer)
        self.requires_feedback_training = True

    def _create_feedback_layer(self, forward_layer: nn.Module) -> nn.Module:
        """Creates a feedback layer matching the forward layer architecture."""
        if isinstance(forward_layer, nn.Linear):
            feedback = nn.Linear(forward_layer.out_features,
                                 forward_layer.in_features,
                                 bias=True)
            # Initialize feedback weights
            with torch.no_grad():
                feedback.weight.copy_(forward_layer.weight.t())
            return feedback
        else:
            self.requires_feedback_training = False
            return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer."""
        # Detach input to prevent global gradient flow
        x = x.detach()
        return self.forward_layer(x)

    def feedback(self, x: torch.Tensor) -> torch.Tensor:
        """Feedback pass through the layer."""
        # Detach input to prevent global gradient flow
        x = x.detach()
        return self.feedback_layer(x)

    def compute_target(self,
                       current_activation: torch.Tensor,
                       next_layer_target: torch.Tensor,
                       next_layer_activation: torch.Tensor) -> torch.Tensor:
        """Compute layer target using difference target propagation."""
        # All inputs should be detached to ensure local training
        current_activation = current_activation.detach()
        next_layer_target = next_layer_target.detach()
        next_layer_activation = next_layer_activation.detach()

        # Apply difference correction
        target = self.feedback(next_layer_target) - \
                 (self.feedback(next_layer_activation) - current_activation)
        return target


class DTPNetwork(nn.Module):
    """Neural network using DTP layers."""

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.dtp_layers = nn.ModuleList([DTPLayer(layer) for layer in layers])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning both output and intermediate activations."""
        activations = [x]
        current = x
        for layer in self.dtp_layers:
            current = layer(current)
            activations.append(current)
        return current, activations

    def compute_targets(self,
                        activations: List[torch.Tensor],
                        final_target: torch.Tensor) -> List[torch.Tensor]:
        """Compute layer-wise targets using DTP."""
        targets = [None] * len(activations)
        targets[-1] = final_target

        # Compute targets for each layer moving backwards
        for i in reversed(range(len(self.dtp_layers))):
            if i == 0:  # No target needed for input layer
                continue

            targets[i] = self.dtp_layers[i].compute_target(
                activations[i],
                targets[i + 1],
                activations[i + 1]
            )

        return targets


class DTPLoss:
    """Loss functions for training DTP networks with difference reconstruction loss."""

    def __init__(self, config: DTPLossConfig):
        self.config = config

    def feedback_loss(self,
                      layer: DTPLayer,
                      input: torch.Tensor,
                      output: torch.Tensor) -> torch.Tensor:
        """
        Compute feedback training loss for a single layer using difference reconstruction loss.
        Since gradient control is handled at layer level, this focuses purely on the DRL calculation.
        """
        if not layer.requires_feedback_training:
            return torch.tensor(0.0, device=input.device)

        # Generate noise samples
        noise_input = torch.randn_like(input) * self.config.noise_scale
        noise_output = torch.randn_like(output) * self.config.noise_scale

        # Forward pass with noise
        noisy_input = input + noise_input

        # Get baseline reconstruction
        baseline_reconstruction = layer.feedback_layer(output)

        # Reconstruction through forward-feedback loop
        reconstruction_noise = layer.feedback_layer(
            layer.forward_layer(noisy_input)
        )

        # Output perturbation reconstruction
        perturbed_output = output + noise_output
        reconstruction_output = layer.feedback_layer(perturbed_output)

        # First term: input perturbation loss
        first_term = -2 * (noise_input * (reconstruction_noise - baseline_reconstruction)).sum(1).mean()

        # Second term: output perturbation loss
        second_term = torch.square(reconstruction_output - baseline_reconstruction).sum(1).mean()

        return first_term + second_term

    def forward_loss(self,
                     network: DTPNetwork,
                     input: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
        """
        Compute forward training loss with local targets.
        Gradient control is now handled by the layers, simplifying the loss calculation.
        """
        # Forward pass
        output, activations = network(input)

        # Initial MSE loss
        mse_loss = 0.5 * ((output - target) ** 2).sum(dim=1).mean()
        grad = torch.autograd.grad(mse_loss,
                                   output,
                                   only_inputs=True,
                                   create_graph=False)[0]

        # Compute output target
        output_target = output - self.config.beta * grad

        # Get targets for all layers
        targets = network.compute_targets(activations, output_target)

        # Calculate layer-wise losses
        total_loss = torch.tensor(0.0, device=input.device)
        for i, layer in enumerate(network.dtp_layers[:-1]):  # Skip last layer
            layer_output = activations[i + 1]
            layer_target = targets[i + 1]

            # Local loss for this layer
            layer_loss = 0.5 * ((layer_output - layer_target) ** 2).sum(1).mean()
            total_loss = total_loss + layer_loss

        return total_loss

    def train_feedback_weights(self,
                               layer: DTPLayer,
                               optimizer: torch.optim.Optimizer,
                               input: torch.Tensor,
                               output: torch.Tensor) -> float:
        """
        Train feedback weights for a single layer over multiple iterations.
        Returns the final loss value.
        """
        final_loss = 0.0
        for i in range(self.config.K_iterations):
            optimizer.zero_grad()
            loss = self.feedback_loss(layer, input, output)
            # On all but the last iteration, retain the graph
            retain_graph = (i < self.config.K_iterations - 1)
            loss.backward(retain_graph=retain_graph)
            optimizer.step()
            final_loss = loss.item()

            # Clear memory between iterations if needed
            if not retain_graph:
                del loss
        return final_loss

    def train_epoch(self,
                    network: DTPNetwork,
                    forward_optimizer: torch.optim.Optimizer,
                    feedback_optimizers: Dict[int, torch.optim.Optimizer],
                    dataloader: torch.utils.data.DataLoader,
                    device: torch.device) -> Dict[str, float]:
        """
        Train for one epoch, handling both forward and feedback weight updates.
        Returns dictionary of average losses.
        """
        total_forward_loss = 0.0
        total_feedback_loss = 0.0
        num_batches = 0

        for batch_input, batch_target in dataloader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            # Train feedback weights first
            output, activations = network(batch_input)
            for i, layer in enumerate(network.dtp_layers[:-1]):  # Skip last layer
                if layer.requires_feedback_training:
                    loss = self.train_feedback_weights(
                        layer,
                        feedback_optimizers[i],
                        activations[i],
                        activations[i + 1]
                    )
                    total_feedback_loss += loss

            # Then train forward weights
            forward_optimizer.zero_grad()
            forward_loss = self.forward_loss(network, batch_input, batch_target)
            forward_loss.backward()
            forward_optimizer.step()

            total_forward_loss += forward_loss.item()
            num_batches += 1

        return {
            'forward_loss': total_forward_loss / num_batches,
            'feedback_loss': total_feedback_loss / num_batches
        }