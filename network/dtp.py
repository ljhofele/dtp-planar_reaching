import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class DTPLossConfig:
    """Configuration for DTP loss calculation."""
    beta: float = 0.2
    noise_scale: float = 0.1
    feedback_samples: int = 1


class DTPLayer(nn.Module):
    """A layer in the DTP network with both forward and feedback pathways."""

    def __init__(self, forward_layer: nn.Module):
        super().__init__()
        self.forward_layer = forward_layer
        # Initialize feedback layer
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
                        final_target: torch.Tensor) -> List[torch.Tensor]:
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
            G = self.dtp_layers[i].feedback
            targets[i] = h_i + G(t_next) - G(h_next)

        return targets


class DTPLoss:
    """Loss functions for training DTP networks."""

    def __init__(self, config: DTPLossConfig):
        self.config = config

    def feedback_loss(self,
                      layer: DTPLayer,
                      input: torch.Tensor,
                      output: torch.Tensor) -> torch.Tensor:
        """
        Compute feedback training loss for a single layer based on Ernoult et al's implementation.
        """
        if not layer.requires_feedback_training:
            return torch.tensor(0.0, device=input.device)

        # Get the forward and feedback layers
        layer_F = layer.forward_layer
        layer_G = layer.feedback_layer

        # Compute initial reconstruction
        r = layer_G(output)

        # Input perturbation
        dx = self.config.noise_scale * torch.randn_like(input)
        with torch.no_grad():
            y_noise = layer_F(input + dx)
        r_noise = layer_G(y_noise)
        dr = r_noise - r

        # Output perturbation
        dy = self.config.noise_scale * torch.randn_like(output)
        r_noise_y = layer_G(output + dy)
        dr_y = r_noise_y - r

        # Compute loss terms
        dr_loss = -2 * (dx * dr).flatten(1).sum(1).mean()
        dy_loss = (dr_y ** 2).flatten(1).sum(1).mean()

        return dr_loss + dy_loss

    def forward_loss(self,
                     network: DTPNetwork,
                     input: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
        """
        Compute forward training loss with proper initialization and stability measures.
        """
        # Forward pass
        output, activations = network(input)

        # Calculate initial target
        temp_output = output.detach().clone()
        temp_output.requires_grad_(True)

        # Use MSE loss for regression
        mse_loss = 0.5 * ((temp_output - target) ** 2).sum(dim=1).mean()
        grad = torch.autograd.grad(mse_loss,
                                   temp_output,
                                   only_inputs=True,
                                   create_graph=False)[0]

        # Compute first target
        output_target = output.detach() - self.config.beta * grad

        # Compute targets for all layers
        targets = network.compute_targets(activations, output_target)

        # Calculate layer-wise losses with stability measures
        layer_losses = []
        for i, (h, t) in enumerate(zip(activations[1:], targets[1:])):
            if not h.requires_grad:
                h.requires_grad_(True)

            # Calculate loss with numerical stability
            loss = 0.5 * ((h - t) ** 2).view(h.size(0), -1).sum(1).mean()  # Sum over features, mean over batch
            loss = loss + 1e-8  # Add small epsilon to prevent exactly zero loss

            layer_losses.append(loss)

        return torch.sum(torch.stack(layer_losses))