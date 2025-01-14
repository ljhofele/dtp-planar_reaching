import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from network.dtp import DTPNetwork, DTPLoss, DTPLossConfig
from environment import create_batch, input_transform, inverse_thetas_transform
from kinematics.planar_arms import PlanarArms


def create_simple_dtp_network(dim_input: int = 4, dim_output: int = 2) -> DTPNetwork:
    layers = [
        nn.Linear(dim_input, 128),  # 4 input features (2 angles in sin + xy target)
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, dim_output)    # 2 output features (delta angles)
    ]

    # Ensure all parameters have requires_grad=True
    for layer in layers:
        if hasattr(layer, 'weight'):
            layer.weight.requires_grad_(True)
        if hasattr(layer, 'bias'):
            layer.bias.requires_grad_(True)

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


def train_epoch(
        network: DTPNetwork,
        loss_fn: DTPLoss,
        forward_optimizer: torch.optim.Optimizer,
        feedback_optimizer: torch.optim.Optimizer,
        num_batches: int,
        batch_size: int,
        arm: str,
        device: torch.device
) -> Dict[str, float]:
    """
    Train the network for one epoch.

    Args:
        network: The DTP network
        loss_fn: The DTP loss function
        forward_optimizer: Optimizer for forward weights
        feedback_optimizer: Optimizer for feedback weights
        num_batches: Number of batches per epoch
        batch_size: Size of each batch
        arm: Which arm to train ('left' or 'right')
        device: Device to train on

    Returns:
        Dictionary containing average losses for the epoch
    """
    network.train()
    total_forward_loss = 0.0
    total_feedback_loss = 0.0

    for _ in range(num_batches):
        # Generate random reaching movements
        inputs, targets = create_batch(
            arm=arm,
            batch_size=batch_size,
            device=device
        )

        # Train feedback weights
        feedback_optimizer.zero_grad()
        total_fb_loss = torch.tensor(0.0, device=device)

        # Forward pass to get activations
        _, activations = network(inputs)

        # Train feedback weights for each layer
        for i, layer in enumerate(network.dtp_layers):
            if i == len(network.dtp_layers) - 1:  # Skip last layer
                continue

            feedback_loss = loss_fn.feedback_loss(
                layer,
                activations[i],
                activations[i + 1]
            )
            total_fb_loss += feedback_loss

        total_fb_loss.backward()
        feedback_optimizer.step()

        # Train forward weights
        forward_optimizer.zero_grad()
        forward_loss = loss_fn.forward_loss(network, inputs, targets)
        forward_loss.backward()
        forward_optimizer.step()

        total_forward_loss += forward_loss.item()
        total_feedback_loss += total_fb_loss.item()

    return {
        'forward_loss': total_forward_loss / num_batches,
        'feedback_loss': total_feedback_loss / num_batches
    }


def train_network(
        network: DTPNetwork,
        loss_fn: DTPLoss,
        num_epochs: int,
        num_batches: int,
        batch_size: int,
        arm: str,
        device: torch.device,
        learning_rate: float = 0.01,
        validation_interval: int = 10
) -> Dict[str, List[float]]:
    """
    Train the network for multiple epochs with validation.

    """
    # Initialize optimizers
    forward_optimizer = torch.optim.SGD(
        network.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=1e-4
    )

    feedback_optimizer = torch.optim.SGD(
        [p for layer in network.dtp_layers
         for p in layer.feedback_layer.parameters()],
        lr=0.05,
        momentum=0.9,
        weight_decay=1e-4
    )

    history = {
        'forward_loss': [],
        'feedback_loss': [],
        'validation_error': []
    }

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Train for one epoch
        epoch_losses = train_epoch(
            network=network,
            loss_fn=loss_fn,
            forward_optimizer=forward_optimizer,
            feedback_optimizer=feedback_optimizer,
            num_batches=num_batches,
            batch_size=batch_size,
            arm=arm,
            device=device
        )

        history['forward_loss'].append(epoch_losses['forward_loss'])
        history['feedback_loss'].append(epoch_losses['feedback_loss'])

        # Run validation periodically
        if (epoch + 1) % validation_interval == 0:
            val_error = evaluate_reaching(
                network=network,
                num_tests=50,
                arm=arm,
                device=device
            )
            history['validation_error'].append(val_error)
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}")
            tqdm.write(f"Forward Loss: {epoch_losses['forward_loss']:.4f}")
            tqdm.write(f"Feedback Loss: {epoch_losses['feedback_loss']:.4f}")
            tqdm.write(f"Validation Error: {val_error:.2f}mm\n")

    return history


def evaluate_reaching(
        network: DTPNetwork,
        num_tests: int,
        arm: str,
        device: torch.device
) -> float:
    """
    Evaluate the network's reaching accuracy.
    """
    network.eval()
    total_error = 0.0

    with torch.no_grad():
        for _ in range(num_tests):
            # Generate a single test movement
            inputs, targets = create_batch(
                arm=arm,
                batch_size=1,
                device=device
            )

            # Get network prediction
            outputs, _ = network(inputs)

            # Convert network outputs back to angles and positions
            target_delta_thetas = targets.cpu().numpy()
            pred_delta_thetas = outputs.cpu().numpy()

            # Calculate reaching error
            current_thetas = inverse_thetas_transform(inputs)
            target_xy = PlanarArms.forward_kinematics(arm=arm,
                                                      thetas=PlanarArms.clip_values(current_thetas[0] + target_delta_thetas[0], radians=True),
                                                      radians=True,
                                                      check_limits=False)[:, -1]
            pred_xy = PlanarArms.forward_kinematics(arm=arm,
                                                    thetas=PlanarArms.clip_values(current_thetas[0] + pred_delta_thetas[0], radians=True),
                                                    radians=True,
                                                    check_limits=False)[:, -1]

            error = np.linalg.norm(target_xy - pred_xy)
            total_error += error

    return total_error / num_tests


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = create_simple_dtp_network().to(device)
    loss_fn = DTPLoss(config=DTPLossConfig())

    # Training
    history = train_network(
        network=network,
        loss_fn=loss_fn,
        num_epochs=100_000,
        num_batches=10,
        batch_size=64,
        arm="right",
        device=device,
        validation_interval=10_000
    )

    # Final evaluation
    final_error = evaluate_reaching(
        network=network,
        num_tests=1000,
        arm="right",
        device=device
    )
    tqdm.write(f"Final reaching error: {final_error:.2f}mm")