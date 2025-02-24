import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from network.dtp import DTPNetwork, DTPConfig
from environment import MovementBuffer, inverse_target_transform, create_batch
from kinematics.planar_arms import PlanarArms


def create_dtp_network(
        layer_dims: List[int],
        activation: str = "tanh",
        config: Optional[DTPConfig] = None,
        bias: bool = True,
        final_activation: Optional[str] = None,
) -> DTPNetwork:
    """Create a DTP network with the specified architecture."""
    if config is None:
        config = DTPConfig()

    return DTPNetwork(
        layer_sizes=layer_dims,
        activation=activation,
        config=config,
        bias=bias,
        final_activation=final_activation
    )


def train_epoch(
        network: DTPNetwork,
        optimizer: torch.optim.Optimizer,
        buffer: MovementBuffer,
        num_batches: int,
        batch_size: int,
) -> Dict[str, float]:
    """Train the network for one epoch using the movement buffer."""
    network.train()
    total_forward_loss = 0.0
    total_feedback_loss = 0.0

    for _ in range(num_batches):
        # Generate a batch
        inputs, targets, _ = buffer.get_batches(batch_size=batch_size)

        # Train step
        optimizer.zero_grad()
        forward_loss, feedback_loss = network.train_step(inputs, targets)

        # Update weights
        forward_loss.backward()
        optimizer.step()

        total_forward_loss += forward_loss.item()
        total_feedback_loss += feedback_loss.item()

    return {
        'forward_loss': total_forward_loss / num_batches,
        'feedback_loss': total_feedback_loss / num_batches
    }


def train_network(
        network: DTPNetwork,
        num_epochs: int,
        num_batches: int,
        batch_size: int,
        trainings_buffer_size: int,
        arm: str,
        device: torch.device,
        learning_rate: float = 1e-4,
        validation_interval: int = 10
) -> Dict[str, List[float]]:
    """Train the network for multiple epochs with validation."""
    # Initialize optimizer
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-5
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
            optimizer=optimizer,
            buffer=trainings_buffer,
            num_batches=num_batches,
            batch_size=batch_size,
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
    """Evaluate the network's reaching accuracy."""
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
            outputs, _ = network.forward(inputs)

            # Convert network outputs and targets back to radians
            target_delta_thetas = inverse_target_transform(targets.cpu().numpy())
            pred_delta_thetas = inverse_target_transform(outputs.cpu().numpy())

            # Calculate reaching error
            target_xy = PlanarArms.forward_kinematics(
                arm=arm,
                thetas=PlanarArms.clip_values(initial_thetas + target_delta_thetas[0], radians=True),
                radians=True,
                check_limits=False
            )[:, -1]
            pred_xy = PlanarArms.forward_kinematics(
                arm=arm,
                thetas=PlanarArms.clip_values(initial_thetas + pred_delta_thetas[0], radians=True),
                radians=True,
                check_limits=False
            )[:, -1]

            error = np.linalg.norm(target_xy - pred_xy)
            total_error += error

    return total_error / num_tests


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', type=str, default="right", choices=["right", "left"])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=5_000)
    parser.add_argument('--trainings_buffer_size', type=int, default=5_000)
    parser.add_argument('--validation_interval', type=int, default=50)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--K_iterations', type=int, default=3, help="Number of feedback updates during training")
    parser.add_argument('--feedback_lr', type=float, default=1e-4)
    parser.add_argument('--plot_history', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print(f'Pytorch version: {torch.__version__} running on {device}')

    # Setup network configuration
    config = DTPConfig(
        beta=args.beta,
        noise_scale=args.noise_scale,
        K_iterations=args.K_iterations,
        learning_rate=args.feedback_lr
    )

    # Create network
    layer_sizes = [4, 128, 128, 2]  # Example sizes for joint angle problem
    network = create_dtp_network(
        layer_dims=layer_sizes,
        activation='elu',
        config=config
    )
    network = network.to(device)

    # Training
    history = train_network(
        network=network,
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        trainings_buffer_size=args.trainings_buffer_size,
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
    print(f"Final reaching error: {final_error:.2f}mm")

    # Plot history
    if args.plot_history:
        import os
        import matplotlib.pyplot as plt

        plot_folder = "figures/"
        os.makedirs(plot_folder, exist_ok=True)

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
        axs[0].plot(history['forward_loss'], label='Forward Loss')
        axs[1].plot(history['feedback_loss'], label='Feedback Loss')
        axs[2].plot(history['validation_error'], label='Validation Error')
        for ax in axs:
            ax.set_xlabel('Epoch')
            ax.legend()
        plt.tight_layout()
        plt.savefig(plot_folder + f"history_{args.arm}.png")
        plt.close(fig)