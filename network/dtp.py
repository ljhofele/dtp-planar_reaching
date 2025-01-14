import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


class DTPLayer(nn.Module):
    """A layer in the DTP network with both forward and feedback pathways."""

    def __init__(self, forward_layer: nn.Module):
        super().__init__()
        self.forward_layer = forward_layer
        # Initialize feedback layer as transpose of forward layer initially
        self.feedback_layer = self._create_feedback_layer(forward_layer)
        self.requires_feedback_training = True

    def _create_feedback_layer(self, forward_layer: nn.Module) -> nn.Module:
        """Creates a feedback layer matching the forward layer architecture."""
        if isinstance(forward_layer, nn.Linear):
            feedback = nn.Linear(forward_layer.out_features, forward_layer.in_features, bias=True)
            with torch.no_grad():
                feedback.weight.copy_(forward_layer.weight.t())
            return feedback
        elif isinstance(forward_layer, nn.Conv2d):
            feedback = nn.ConvTranspose2d(
                forward_layer.out_channels,
                forward_layer.in_channels,
                forward_layer.kernel_size,
                stride=forward_layer.stride,
                padding=forward_layer.padding
            )
            return feedback
        else:
            self.requires_feedback_training = False
            return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer."""
        return self.forward_layer(x)

    def feedback(self, x: torch.Tensor) -> torch.Tensor:
        """Feedback pass through the layer."""
        return self.feedback_layer(x)


class DTPNetwork(nn.Module):
    """Network implementing Difference Target Propagation."""

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.dtp_layers = nn.ModuleList([DTPLayer(layer) for layer in layers])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning both output and all intermediate activations."""
        activations = [x]
        current = x
        for layer in self.dtp_layers:
            current = layer(current)
            activations.append(current)
        return current, activations

    def compute_targets(self,
                        activations: List[torch.Tensor],
                        final_target: torch.Tensor, ) -> List[torch.Tensor]:
        """Compute layer-wise targets using DTP."""
        targets = [None] * len(activations)
        targets[-1] = final_target

        # Compute targets for each layer moving backwards
        for i in reversed(range(len(self.dtp_layers))):
            if i == 0:  # No target needed for input layer
                continue

            # Get activations and target for current layer
            h_i = activations[i]
            t_next = targets[i + 1]
            h_next = activations[i + 1]

            # Compute target using DTP formula
            # t_i = h_i + G(t_next) - G(h_next)
            G = self.dtp_layers[i].feedback
            targets[i] = h_i + G(t_next) - G(h_next)

        return targets


@dataclass
class DTPLossConfig:
    """Configuration for DTP loss calculation."""
    beta: float = 0.1
    noise_scale: float = 0.1
    feedback_samples: int = 1


class DTPLoss:
    """Loss functions for training DTP networks."""

    def __init__(self, config: DTPLossConfig):
        self.config = config

    def feedback_loss(self,
                      layer: DTPLayer,
                      input: torch.Tensor,
                      output: torch.Tensor,  # We are using L-DRL with noisy outputs from the forward pass
                      ) -> torch.Tensor:
        """Compute feedback training loss for a single layer."""
        if not layer.requires_feedback_training:
            return torch.tensor(0.0, device=input.device)

        batch_losses = []
        for _ in range(self.config.feedback_samples):
            # Add noise to input
            noise = torch.randn_like(input) * self.config.noise_scale
            noisy_input = input + noise

            # Forward pass
            noisy_output = layer(noisy_input)

            # Feedback pass
            reconstructed = layer.feedback(noisy_output)

            # L-DRL loss
            reconstruction_error = reconstructed - noisy_input
            loss = torch.mean(0.5 * torch.sum(reconstruction_error ** 2, dim=1) -
                              torch.sum(noise * reconstruction_error, dim=1))

            batch_losses.append(loss)

        return torch.mean(torch.stack(batch_losses))

    def forward_loss(self,
                     network: DTPNetwork,
                     inputs: torch.Tensor,
                     targets: torch.Tensor,) -> torch.Tensor:
        """
        Compute forward training loss for reaching.
        """
        # Forward pass
        with torch.set_grad_enabled(True):
            output, activations = network(inputs)

        # Compute reaching error (L2 norm)
        reaching_error = 0.5 * torch.mean((output - targets) ** 2)

        # Set first target using reaching error gradient
        temp_output = output.detach().clone()
        temp_output.requires_grad_(True)

        # Scale gradients to prevent explosion
        grad_norm = torch.norm(reaching_error)
        if grad_norm > 1.0:
            reaching_error = reaching_error / grad_norm

        output_target = output.detach() - self.config.beta * reaching_error

        # Compute targets for all layers
        targets = network.compute_targets(activations, output_target)

        # Calculate layer-wise losses with stability measures
        layer_losses = []
        for i, (h, t) in enumerate(zip(activations[1:], targets[1:])):
            if not h.requires_grad:
                h.requires_grad_(True)

            # Normalize activations and targets
            h_norm = torch.norm(h)
            t_norm = torch.norm(t)
            if h_norm > 1.0:
                h = h / h_norm
            if t_norm > 1.0:
                t = t / t_norm

            # Calculate loss with numerical stability
            loss = 0.5 * ((h - t) ** 2).view(h.size(0), -1)
            loss = loss.sum(1).mean()  # Sum over features, mean over batch

            # Add small epsilon to prevent exactly zero loss
            loss = loss + 1e-8

            layer_losses.append(loss)

        return torch.sum(torch.stack(layer_losses))

    def forward_loss_ce(self,
                        network: DTPNetwork,
                        input: torch.Tensor,
                        target: torch.Tensor, ) -> torch.Tensor:
        """Compute forward training loss with numerical stability measures."""
        # Forward pass
        with torch.set_grad_enabled(True):
            output, activations = network(input)

        # Calculate initial target with gradient scaling
        temp_output = output.detach().clone()
        temp_output.requires_grad_(True)
        ce_loss = F.cross_entropy(temp_output, target, reduction="sum")
        grad = torch.autograd.grad(
            ce_loss,
            temp_output,
            only_inputs=True,
            create_graph=False
        )[0]

        # Scale gradients to prevent explosion
        grad_norm = torch.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm

        output_target = output.detach() + (-self.config.beta * grad)

        # Compute targets for all layers
        targets = network.compute_targets(activations, output_target)

        # Calculate layer-wise losses with stability measures
        layer_losses = []
        for i, (h, t) in enumerate(zip(activations[1:], targets[1:])):
            if not h.requires_grad:
                h.requires_grad_(True)

            # Normalize activations and targets
            h_norm = torch.norm(h)
            t_norm = torch.norm(t)
            if h_norm > 1.0:
                h = h / h_norm
            if t_norm > 1.0:
                t = t / t_norm

            # Calculate loss with numerical stability
            loss = 0.5 * ((h - t) ** 2).view(h.size(0), -1)
            loss = loss.sum(1).mean()  # Sum over features, mean over batch

            # Add small epsilon to prevent exactly zero loss
            loss = loss + 1e-8

            layer_losses.append(loss)

        return torch.sum(torch.stack(layer_losses))
