import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DTPLossConfig:
    """Configuration for DTP loss calculation."""
    beta: float = 0.9  # Damping factor for target propagation
    noise_scale: float = 0.1  # Scale of noise for feedback training


class DTPLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 forward_requires_grad: bool = False,
                 activation: str = 'elu'):
        super().__init__()

        # Store activation type first
        self._activation = activation

        # Forward parameters:
        self._weights = nn.Parameter(torch.Tensor(out_features, in_features),
                                     requires_grad=forward_requires_grad)
        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_features),
                                      requires_grad=forward_requires_grad)
        else:
            self._bias = None

        self._feedback_weights = nn.Parameter(torch.Tensor(in_features, out_features), requires_grad=False)

        if bias:
            self._feedback_bias = nn.Parameter(torch.Tensor(in_features),
                                               requires_grad=False)
        else:
            self._feedback_bias = None

        # Cache for activations
        self._activations = None  # Post-activation
        self._linear_activations = None  # Pre-activation
        self._target = None
        self._reconstruction_loss = None

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights and biases."""
        # Calculate fan in/out for Xavier/Kaiming init
        fan_in = self._weights.size(1)
        fan_out = self._weights.size(0)

        # Initialize forward weights based on activation
        if self._activation in ['linear', 'sigmoid', 'tanh']:
            # Xavier initialization
            bound = np.sqrt(6. / (fan_in + fan_out))
            nn.init.uniform_(self._weights, -bound, bound)
        else:
            # Kaiming initialization for ReLU-like functions
            bound = np.sqrt(6. / fan_in)  # He initialization bound
            nn.init.uniform_(self._weights, -bound, bound)

        if self._bias is not None:
            # Initialize bias to small values
            nn.init.uniform_(self._bias, -0.1, 0.1)

        # Initialize feedback weights with same scheme as forward weights
        if self._activation in ['linear', 'sigmoid', 'tanh']:
            bound = np.sqrt(6. / (fan_in + fan_out))
            nn.init.uniform_(self._feedback_weights, -bound, bound)
        else:
            bound = np.sqrt(6. / fan_out)  # Note: fan_out for feedback is fan_in for forward
            nn.init.uniform_(self._feedback_weights, -bound, bound)

        if self._feedback_bias is not None:
            nn.init.zeros_(self._feedback_bias)

    def propagate_backward(self, h: torch.Tensor) -> torch.Tensor:
        """Propagate activations backward through feedback weights."""
        print(h.shape, self._feedback_weights.shape)
        # Note: need to transpose feedback weights for proper dimensions
        result = F.linear(h, self._feedback_weights, self._feedback_bias)
        return result

    def get_forward_parameter_list(self):
        """Return list of forward parameters."""
        params = [self._weights]
        if self._bias is not None:
            params.append(self._bias)
        return params

    def get_feedback_parameter_list(self):
        """Return list of feedback parameters."""
        params = [self._feedback_weights]
        if self._feedback_bias is not None:
            params.append(self._feedback_bias)
        return params

    def forward_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward activation function."""
        if self._activation == 'linear':
            return x
        elif self._activation == 'tanh':
            return torch.tanh(x)
        elif self._activation == 'relu':
            return F.relu(x)
        elif self._activation == 'elu':
            return F.elu(x)
        elif self._activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError(f'Activation function {self._activation} not supported')

    def compute_jacobian_approximation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute approximation of the Jacobian matrix for the activation function."""
        if self._activation == 'linear':
            return torch.ones_like(x)
        elif self._activation == 'tanh':
            return 1 - torch.tanh(x) ** 2
        elif self._activation == 'relu':
            return (x > 0).float()
        elif self._activation == 'elu':
            # For ELU: f'(x) = 1 if x > 0 else α * exp(x)
            alpha = 1.0  # Default ELU parameter
            jac = torch.ones_like(x)
            mask = x <= 0
            jac[mask] = alpha * torch.exp(x[mask])
            return jac
        elif self._activation == 'sigmoid':
            sigmoid_x = torch.sigmoid(x)
            return sigmoid_x * (1 - sigmoid_x)
        else:
            raise ValueError(f'Activation function {self._activation} not supported')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer."""
        # Linear transformation
        self._linear_activations = F.linear(x, self._weights, self._bias)

        # Activation
        self._activations = self.forward_activation(self._linear_activations)

        return self._activations

    def backward(self,
                output_target: torch.Tensor,
                layer_activation: torch.Tensor,
                output_activation: torch.Tensor) -> torch.Tensor:
        """Compute target for this layer using difference target propagation."""
        # Get reconstructed activations using feedback weights
        target_reconstruction = self.propagate_backward(output_target)  # [batch_size, layer_dim]
        activation_reconstruction = self.propagate_backward(output_activation)  # [batch_size, layer_dim]

        print(target_reconstruction.shape, layer_activation.shape, activation_reconstruction.shape)
        # Apply difference correction formula from DTP paper:
        # h_target = g(h_target_next) + (h - g(h_next))
        layer_target = target_reconstruction + (layer_activation - activation_reconstruction)

        return layer_target

    def compute_forward_gradients(self,
                                  h_target: torch.Tensor,
                                  h_previous: torch.Tensor,
                                  norm_ratio: float = 1.):
        """Compute gradients for the forward parameters."""
        # Get Jacobian approximation for activation function
        if self._activation == 'linear':
            teaching_signal = 2 * (self.activations - h_target)
        else:
            jacobian = self.compute_jacobian_approximation(self._linear_activations)
            teaching_signal = 2 * jacobian * (self.activations - h_target)

        # Compute gradients
        batch_size = h_target.shape[0]

        # Bias gradient
        if self._bias is not None:
            bias_grad = teaching_signal.mean(0)
            self._bias.grad = bias_grad.detach()

        # Weight gradient
        weights_grad = 1. / batch_size * teaching_signal.t().mm(h_previous)
        self._weights.grad = weights_grad.detach()

    def compute_feedback_gradients(self,
                                   h_corrupted: torch.Tensor,
                                   output_corrupted: torch.Tensor,
                                   output_activation: torch.Tensor,
                                   sigma: float):
        """Compute gradients for feedback parameters using difference reconstruction loss."""
        self.set_feedback_requires_grad(True)

        # Get reconstructed activations
        h_reconstructed = self.backward(output_corrupted,
                                        self._activations,
                                        output_activation)

        # Compute reconstruction loss
        if sigma <= 0:
            raise ValueError('Sigma must be positive for difference reconstruction loss')

        scale = 1 / sigma ** 2
        reconstruction_loss = scale * F.mse_loss(h_reconstructed, h_corrupted)

        # Save reconstruction loss
        self._reconstruction_loss = reconstruction_loss.item()

        # Compute gradients
        grads = torch.autograd.grad(reconstruction_loss,
                                    self.get_feedback_parameter_list(),
                                    retain_graph=False)

        for param, grad in zip(self.get_feedback_parameter_list(), grads):
            param.grad = grad.detach()

        self.set_feedback_requires_grad(False)

    def set_feedback_requires_grad(self, value: bool):
        """Set requires_grad for feedback parameters."""
        self._feedback_weights.requires_grad = value
        if self._feedback_bias is not None:
            self._feedback_bias.requires_grad = value

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def feedback_weights(self):
        return self._feedback_weights

    @property
    def feedback_bias(self):
        return self._feedback_bias

    @property
    def activations(self):
        return self._activations

    @property
    def linear_activations(self):
        return self._linear_activations

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value


class DTPNetwork(nn.Module):
    """Network implementing Difference Target Propagation."""

    def __init__(self,
                 layer_dims: List[int],
                 activation: str = 'elu',
                 bias: bool = True):
        super().__init__()

        self.layers = nn.ModuleList()
        print(f"Creating network with dimensions: {layer_dims}")

        # Create layers
        for i in range(len(layer_dims) - 1):
            # Use linear activation for output layer
            current_activation = 'linear' if i == len(layer_dims) - 2 else activation

            layer = DTPLayer(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                bias=bias,
                activation=current_activation
            )
            print(f"Layer {i}: {layer_dims[i]} -> {layer_dims[i + 1]} ({current_activation})")
            self.layers.append(layer)

        self._input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        self._input = x

        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def train_feedback_weights(self, sigma: float):
        """Train feedback weights for all layers."""
        # Skip output layer in feedback training
        for i in range(len(self.layers) - 1, 0, -1):
            # Get current layer activations
            h_current = self.layers[i].activations

            # Add noise to current layer activations
            h_corrupted = h_current + sigma * torch.randn_like(h_current)

            # Forward pass starting from corrupted activations
            output_corrupted = h_corrupted
            for j in range(i + 1, len(self.layers)):
                output_corrupted = self.layers[j](output_corrupted)

            # Get real output activations
            output_activation = self.layers[-1].activations

            # Compute feedback gradients
            self.layers[i].compute_feedback_gradients(
                h_corrupted=h_corrupted,
                output_corrupted=output_corrupted,
                output_activation=output_activation,
                sigma=sigma
            )

    def backward(self, loss: torch.Tensor, target_lr: float, save_target: bool = False):
        """Backward pass to compute targets and update parameters."""
        # Get output activation
        output_activation = self.layers[-1].activations

        # Get gradient of loss w.r.t output activation
        gradient = torch.autograd.grad(loss, output_activation, retain_graph=True)[0].detach()

        # Compute output target
        output_target = output_activation - target_lr * gradient

        if save_target:
            self.layers[-1].target = output_target

        # Update output layer
        self.layers[-1].compute_forward_gradients(
            output_target,
            self.layers[-2].activations
        )

        # Propagate targets backward
        for i in range(len(self.layers) - 2, 0, -1):
            # Get previous activation (input for first layer)
            h_previous = self._input if i == 0 else self.layers[i - 1].activations

            # Compute target for current layer
            h_target = self.layers[i].backward(
                output_target=output_target,
                layer_activation=self.layers[i].activations,
                output_activation=output_activation
            )

            if save_target:
                self.layers[i].target = h_target

            # Compute gradients
            self.layers[i].compute_forward_gradients(
                h_target=h_target,
                h_previous=h_previous
            )

    def get_forward_parameter_list(self):
        """Get list of forward parameters."""
        params = []
        for layer in self.layers:
            params.extend(layer.get_forward_parameter_list())
        return params

    def get_feedback_parameter_list(self):
        """Get list of feedback parameters."""
        params = []
        for layer in self.layers[:-1]:  # Skip output layer
            params.extend(layer.get_feedback_parameter_list())
        return params
