import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from network.dtp import DTPNetwork, DTPLoss, DTPLossConfig


def create_simple_dtp_network():
    layers = [
        nn.Linear(6, 128),  # 6 input features (2 angles in sin/cos + 2D target)
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 6)    # 6 output features (delta angles + target position)
    ]
    return DTPNetwork(layers)


def create_dtp_network(
        layer_dims: List[int],
        activation_fn: Optional[str] = "relu",
        final_activation: Optional[str] = None
) -> DTPNetwork:
    """
    Create a DTP network with customizable architecture.
    """
    # Dictionary mapping activation names to functions
    activation_fns = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'elu': nn.ELU(),
        None: nn.Identity()
    }

    if activation_fn not in activation_fns:
        raise ValueError(f"Unsupported activation function: {activation_fn}. "
                         f"Choose from {list(activation_fns.keys())}")

    if final_activation not in activation_fns:
        raise ValueError(f"Unsupported final activation: {final_activation}. "
                         f"Choose from {list(activation_fns.keys())}")

    layers = []

    # Create layers with specified dimensions and activations
    for i in range(len(layer_dims) - 1):
        # Add linear layer
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

        # Add activation function
        if i < len(layer_dims) - 2:  # Not last layer
            if activation_fn:
                layers.append(activation_fns[activation_fn])
        else:  # Last layer
            if final_activation:
                layers.append(activation_fns[final_activation])

    return DTPNetwork(layers)


def train_step(network: DTPNetwork,
               loss_fn: DTPLoss,
               input: torch.Tensor,
               target: torch.Tensor,
               forward_optimizer: torch.optim.Optimizer,
               feedback_optimizer: torch.optim.Optimizer):
    """Single training step for DTP."""
    # Train feedback weights
    feedback_optimizer.zero_grad()
    total_feedback_loss = torch.tensor(0.0, device=input.device)

    # Forward pass to get activations
    _, activations = network(input)

    # Train feedback weights for each layer
    for i, layer in enumerate(network.dtp_layers):
        if i == len(network.dtp_layers) - 1:  # Skip last layer
            continue

        feedback_loss = loss_fn.feedback_loss(
            layer,
            activations[i],
            activations[i + 1]
        )
        total_feedback_loss += feedback_loss

    total_feedback_loss.backward()
    feedback_optimizer.step()

    # Train forward weights
    forward_optimizer.zero_grad()
    forward_loss = loss_fn.forward_loss(network, input, target)
    forward_loss.backward()
    forward_optimizer.step()

    return forward_loss.item(), total_feedback_loss.item()


if __name__ == "__main__":
    from environment import create_batch
    # parameters
    batch_size = 64
    num_epochs = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create network and loss function
    network = create_simple_dtp_network()
    network = network.float()  # Ensure network is in float32
    loss_fn = DTPLoss(DTPLossConfig())

    # Create optimizers
    forward_optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
    feedback_optimizer = torch.optim.SGD(
        [p for layer in network.dtp_layers
         for p in layer.feedback_layer.parameters()],
        lr=0.01
    )

    for _ in range(num_epochs):
        inputs, targets = create_batch(batch_size=batch_size,
                                       arm="right",
                                       device=device)

        # ensure inputs and targets are on the same device as the network
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)

        f_loss, b_loss = train_step(
            network=network,
            loss_fn=loss_fn,
            input=inputs,
            target=targets,
            forward_optimizer=forward_optimizer,
            feedback_optimizer=feedback_optimizer
        )

        print(f"Forward loss: {f_loss}, Feedback loss: {b_loss}")