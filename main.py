import torch
import numpy as np
from tqdm import tqdm
from network.dtp import DTPNetwork
from environment import MovementBuffer, inverse_target_transform, create_batch
from kinematics.planar_arms import PlanarArms


def train_network(
        network: DTPNetwork,
        num_epochs: int,
        num_batches: int,
        batch_size: int,
        trainings_buffer_size: int,
        arm: str,
        device: torch.device,
        learning_rate: float = 1e-4,
        learning_rate_fb: float = 1e-4,
        target_lr: float = 0.1,
        sigma: float = 0.1,
        validation_interval: int = 10
) -> dict:
    """
    Train the network using DTP.
    """
    # Initialize optimizers
    forward_optimizer = torch.optim.Adam(
        network.get_forward_parameter_list(),
        lr=learning_rate
    )

    feedback_optimizer = torch.optim.Adam(
        network.get_feedback_parameter_list(),
        lr=learning_rate_fb
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
            forward_optimizer=forward_optimizer,
            feedback_optimizer=feedback_optimizer,
            buffer=trainings_buffer,
            num_batches=num_batches,
            batch_size=batch_size,
            target_lr=target_lr,
            sigma=sigma
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


def train_epoch(
        network: DTPNetwork,
        forward_optimizer: torch.optim.Optimizer,
        feedback_optimizer: torch.optim.Optimizer,
        buffer: MovementBuffer,
        num_batches: int,
        batch_size: int,
        target_lr: float,
        sigma: float,
) -> dict:
    """Train network for one epoch."""

    network.train()
    total_forward_loss = 0.0
    total_feedback_loss = 0.0

    for _ in range(num_batches):
        # Get batch
        inputs, targets, _ = buffer.get_batches(batch_size=batch_size)

        # Forward pass
        predictions = network(inputs)
        forward_loss = torch.nn.functional.mse_loss(predictions, targets)

        # Train feedback weights
        print('Training feedback weights')
        feedback_optimizer.zero_grad()
        network.train_feedback_weights(sigma=sigma)
        feedback_optimizer.step()

        # Train forward weights using DTP
        print('Training forward weights')
        forward_optimizer.zero_grad()
        network.backward(forward_loss, target_lr=target_lr)
        forward_optimizer.step()

        total_forward_loss += forward_loss.item()
        # Get average reconstruction loss across layers as feedback loss
        feedback_loss = np.mean([layer._reconstruction_loss for layer in network.layers[:-1]
                                 if layer._reconstruction_loss is not None])
        total_feedback_loss += feedback_loss

    return {
        'forward_loss': total_forward_loss / num_batches,
        'feedback_loss': total_feedback_loss / num_batches
    }


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
            # Generate test movement
            inputs, targets, initial_thetas = create_batch(
                arm=arm,
                device=device
            )

            # Get network prediction
            predictions = network(inputs)

            # Convert predictions and targets back to angles
            target_delta_thetas = inverse_target_transform(targets.cpu().numpy())
            pred_delta_thetas = inverse_target_transform(predictions.cpu().numpy())

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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10_000)
    parser.add_argument('--trainings_buffer_size', type=int, default=5_000)
    parser.add_argument('--validation_interval', type=int, default=1000)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_fb', type=float, default=1e-4)
    parser.add_argument('--target_lr', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print(f'Pytorch version: {torch.__version__} running on {device}')

    # Setup network
    network = DTPNetwork(
        layer_dims=[4, 128, 64, 2],
        activation='elu',
        bias=True
    ).to(device)

    # Training
    history = train_network(
        network=network,
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        trainings_buffer_size=args.trainings_buffer_size,
        arm=args.arm,
        device=device,
        learning_rate=args.lr,
        learning_rate_fb=args.lr_fb,
        target_lr=args.target_lr,
        sigma=args.sigma,
        validation_interval=args.validation_interval
    )

    # Final evaluation
    final_error = evaluate_reaching(
        network=network,
        num_tests=1_000,
        arm=args.arm,
        device=device
    )
    print(f"Final reaching error: {final_error:.2f}mm")
