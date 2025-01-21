import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from network.dtp import DTPNetwork, DTPLoss, DTPLossConfig
from environment import MovementBuffer, inverse_target_transform, create_batch
from kinematics.planar_arms import PlanarArms


def create_dtp_network(
        layer_dims: List[int],
        activation_fn: Optional[str] = "relu"
) -> DTPNetwork:
    """Create a DTP network."""

    activation_fns = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.1),  # Better for gradient flow
        'elu': nn.ELU(),
        None: nn.Identity()
    }

    if activation_fn not in activation_fns:
        raise ValueError(f"Unsupported activation function: {activation_fn}")

    layers = []

    # Create layers with kaiming initialization
    for i in range(len(layer_dims) - 1):
        linear = nn.Linear(layer_dims[i], layer_dims[i + 1])
        nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
        nn.init.zeros_(linear.bias)
        layers.append(linear)

        # Add activation except for last layer
        if i < len(layer_dims) - 2 and activation_fn:
            layers.append(activation_fns[activation_fn])

    return DTPNetwork(layers)


def train_epoch(
        network: DTPNetwork,
        loss_fn: DTPLoss,
        forward_optimizer: torch.optim.Optimizer,
        feedback_optimizer: torch.optim.Optimizer,
        buffer: MovementBuffer,
        num_batches: int,
        batch_size: int,
        K_updates: int = 1,
        device: torch.device = torch.device('cpu'),
) -> Dict[str, float]:
    """
    Train the network for one epoch.
    """
    network.train()
    total_forward_loss = 0.0
    total_feedback_loss = 0.0

    for _ in range(num_batches):
        # Generate a single batch
        inputs, targets, _ = buffer.get_batches(batch_size=batch_size)

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
        trainings_buffer_size: int,
        arm: str,
        device: torch.device,
        learning_rate: float = 1e-4,
        validation_interval: int = 10
) -> Dict[str, List[float]]:
    """
    Train the network for multiple epochs with validation.

    """
    # Initialize optimizers
    forward_optimizer = torch.optim.SGD(
        network.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )

    feedback_optimizer = torch.optim.SGD(
        [p for layer in network.dtp_layers
         for p in layer.feedback_layer.parameters()],
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )

    # Initialize dataset
    trainings_buffer = MovementBuffer(
        arm=arm,
        buffer_size=trainings_buffer_size,
        device=device
    )

    # Initialize history
    history = {
        'forward_loss': [],
        'feedback_loss': [],
        'validation_error': []
    }

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Fill buffer with movements
        trainings_buffer.fill_buffer()

        # Train for one epoch
        epoch_losses = train_epoch(
            network=network,
            loss_fn=loss_fn,
            forward_optimizer=forward_optimizer,
            feedback_optimizer=feedback_optimizer,
            buffer=trainings_buffer,
            num_batches=num_batches,
            batch_size=batch_size,
            device=device
        )

        history['forward_loss'].append(epoch_losses['forward_loss'])
        history['feedback_loss'].append(epoch_losses['feedback_loss'])

        # Clear buffer
        trainings_buffer.clear_buffer()

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
            inputs, targets, initial_thetas = create_batch(
                arm=arm,
                device=device
            )

            # Get network prediction
            outputs, _ = network(inputs)

            # Convert network outputs and targets back to radians
            target_delta_thetas = inverse_target_transform(targets.cpu().numpy())
            pred_delta_thetas = inverse_target_transform(outputs.cpu().numpy())

            # Calculate reaching error
            target_xy = PlanarArms.forward_kinematics(arm=arm,
                                                      thetas=PlanarArms.clip_values(initial_thetas + target_delta_thetas[0], radians=True),
                                                      radians=True,
                                                      check_limits=False)[:, -1]
            pred_xy = PlanarArms.forward_kinematics(arm=arm,
                                                    thetas=PlanarArms.clip_values(initial_thetas + pred_delta_thetas[0], radians=True),
                                                    radians=True,
                                                    check_limits=False)[:, -1]

            error = np.linalg.norm(target_xy - pred_xy)
            total_error += error

    return total_error / num_tests


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', type=str, default="right"
                        , choices=["right", "left"])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10_000)
    parser.add_argument('--trainings_buffer_size', type=int, default=5_000)
    parser.add_argument('--validation_interval', type=int, default=1000)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print(f'Pytorch version: {torch.__version__} running on {device}')

    # Setup
    network = create_dtp_network(
        layer_dims=[4, 128, 64, 2],
        activation_fn="leaky_relu").to(device)
    loss_fn = DTPLoss(config=DTPLossConfig(beta=args.beta, noise_scale=args.noise_scale))

    # Training
    history = train_network(
        network=network,
        loss_fn=loss_fn,
        trainings_buffer_size=args.trainings_buffer_size,
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        arm=args.arm,
        device=device,
        validation_interval=args.validation_interval,
        learning_rate=args.lr
    )

    # Final evaluation
    final_error = evaluate_reaching(
        network=network,
        num_tests=1_000,
        arm=args.arm,
        device=device
    )
    tqdm.write(f"Final reaching error: {final_error:.2f}mm")