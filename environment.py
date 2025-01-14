import numpy as np
import torch
from typing import Tuple, Optional
from torch import Tensor
from kinematics.planar_arms import PlanarArms


def input_transform(thetas: np.ndarray,
                    xy: np.ndarray) -> np.ndarray:
    """
    Transforms angles in radians to normalized values and concatenates with normalized xy coordinates
    For both joints, we normalize using the same limits since they're identical!!! If they're different, we need to
    normalize them separately

    :param thetas: initial joint angles in radians (batch_size, 2)
    :param xy: target cartesian coordinates (batch_size, 2)
    :return: normalized inputs in range [-1, 1] (batch_size, 4)
    """
    # For both joints, we normalize using the same limits since they're identical
    angles_normalized = 2 * (thetas - PlanarArms.l_upper_arm_limit) / (
                PlanarArms.u_upper_arm_limit - PlanarArms.l_upper_arm_limit) - 1

    # Normalize xy coordinates using existing function
    xy_normalized = norm_xy(xy)

    return np.concatenate((angles_normalized, xy_normalized), axis=1)


def inverse_input_transform(inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms normalized inputs back to original space
    :param inputs: normalized inputs in range [-1, 1] (batch_size, 4)
    :return: tuple of (joint angles in radians, xy coordinates)
    """
    # Split inputs into angles and coordinates
    angles_norm = inputs[:, :2]
    xy_norm = inputs[:, 2:]

    # Convert normalized angles back to radians
    thetas = (angles_norm + 1) / 2 * (
                PlanarArms.u_upper_arm_limit - PlanarArms.l_upper_arm_limit) + PlanarArms.l_upper_arm_limit

    # Denormalize xy coordinates
    x_mid = (PlanarArms.x_limits[0] + PlanarArms.x_limits[1]) / 2
    y_mid = (PlanarArms.y_limits[0] + PlanarArms.y_limits[1]) / 2
    x_half_range = (PlanarArms.x_limits[1] - PlanarArms.x_limits[0]) / 2
    y_half_range = (PlanarArms.y_limits[1] - PlanarArms.y_limits[0]) / 2

    x = xy_norm[:, 0] * x_half_range + x_mid
    y = xy_norm[:, 1] * y_half_range + y_mid
    xy = np.stack([x, y], axis=1)

    return thetas, xy


def target_transform(thetas: np.ndarray) -> np.ndarray:
    """
    Transforms angle changes to normalized range [-1, 1]
    :param thetas: angle changes in radians (batch_size, 2)
    :return: normalized angle changes (batch_size, 2)
    """
    # Maximum angle change is the full range (≈ 3.14 radians or 180 degrees)
    max_angle_change = PlanarArms.u_upper_arm_limit - PlanarArms.l_upper_arm_limit

    # Normalize the changes to [-1, 1]
    return thetas / (max_angle_change / 2)


def inverse_target_transform(normalized_thetas: np.ndarray) -> np.ndarray:
    """
    Transforms normalized angle changes back to radians
    :param normalized_thetas: normalized angle changes in [-1, 1] range (batch_size, 2)
    :return: angle changes in radians (batch_size, 2)
    """
    # Maximum angle change
    max_angle_change = PlanarArms.u_upper_arm_limit - PlanarArms.l_upper_arm_limit

    # Convert back to radians
    return normalized_thetas * (max_angle_change / 2)


def create_batch(
        arm: str,
        batch_size: int,
        device: torch.device) -> Tuple[Tensor, Tensor, np.ndarray]:
    """Creates a batch of inputs and targets from a random initial position
    :param arm: Right or left arm
    :param batch_size: Number of random points to generate
    :param device: Device to train on
    :return: Tuple of (inputs, targets) and initial joint angles"""

    # Generate random initial joint angles
    init_shoulder_theta = np.random.uniform(low=PlanarArms.l_upper_arm_limit, high=PlanarArms.u_upper_arm_limit)
    init_elbow_theta = np.random.uniform(low=PlanarArms.l_forearm_limit, high=PlanarArms.u_forearm_limit)
    init_thetas = np.array((init_shoulder_theta, init_elbow_theta))

    # Generate random reaching points from initial position
    delta_thetas, targets_xy = generate_random_coordinates(arm=arm,
                                                           initial_thetas=init_thetas,
                                                           num_movements=batch_size,
                                                           return_thetas_radians=True)

    # input for network (init_angles, target_position [in cartesian coordinates])
    batched_init_thetas = np.repeat(init_thetas, batch_size).reshape((batch_size, -1))
    input_batch = input_transform(thetas=batched_init_thetas,
                                  xy=targets_xy)

    target_batch = target_transform(thetas=delta_thetas)
    return (torch.from_numpy(input_batch).float().to(device),  # input (batch_size, 4)  -> initial state and goal
            torch.from_numpy(target_batch).float().to(device),  # target (batch_size, 2) -> action to reach target xy from initial position
            init_thetas)  # initial joint angles in radians


def generate_random_coordinates(arm: str,
                                initial_thetas: np.ndarray,  # in radians
                                num_movements: int,
                                min_distance: float = 50.,
                                return_thetas_radians: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random reaching points that are at least min_distance away from initial position.
    :param arm: Right or left arm
    :param initial_thetas: Initial joint angles in radians
    :param num_movements: Number of random points to generate
    :param x_bounds: X-axis bounds for the workspace
    :param y_bounds: Y-axis bounds for the workspace
    :param min_distance: Minimum distance required from initial position
    :param return_thetas_radians: Whether to return angles in radians

    :return: Tuple of (joint angle changes to the target, target in cartesian coordinates)
    """
    lower_arm_limits, upper_arm_limits = PlanarArms.get_bounds()
    init_pos = PlanarArms.forward_kinematics(arm=arm,
                                             thetas=initial_thetas,
                                             radians=False)[:, -1]

    random_movement_thetas = []
    random_target_xy = []

    while len(random_movement_thetas) < num_movements:
        # Generate random joint angles within limits
        random_thetas = np.random.uniform(lower_arm_limits, upper_arm_limits)

        # Calculate end effector position for these angles
        current_pos = PlanarArms.forward_kinematics(arm=arm,
                                                    thetas=random_thetas,
                                                    radians=True)[:, -1]

        # Calculate distance from initial position
        distance = np.linalg.norm(current_pos - init_pos)

        if distance >= min_distance:
            # Calculate change in joint angles
            theta_change = random_thetas - initial_thetas
            if not return_thetas_radians:
                theta_change = np.degrees(theta_change)

            # Normalize the xy coordinates
            random_movement_thetas.append(theta_change)
            random_target_xy.append(current_pos)

    return np.array(random_movement_thetas), np.array(random_target_xy)


def norm_xy(xy: np.ndarray,
            x_bounds: Tuple[float, float] = PlanarArms.x_limits,
            y_bounds: Tuple[float, float] = PlanarArms.y_limits,) -> np.ndarray:
    # Calculate the midpoints of x and y ranges
    x_mid = (x_bounds[0] + x_bounds[1]) / 2
    y_mid = (y_bounds[0] + y_bounds[1]) / 2

    # Calculate the half-ranges
    x_half_range = (x_bounds[1] - x_bounds[0]) / 2
    y_half_range = (y_bounds[1] - y_bounds[0]) / 2

    # Normalize to [-1, 1]
    normalized_x = (xy[:, 0] - x_mid) / x_half_range
    normalized_y = (xy[:, 1] - y_mid) / y_half_range

    return np.stack([normalized_x, normalized_y], axis=1)


if __name__ == "__main__":
    thetas, xys = generate_random_coordinates(arm='right',
                                              initial_thetas=np.radians([90., 90.]),
                                              num_movements=100,)

    print(thetas)
    print(xys)

    inputs, targets, _ = create_batch(arm='right', batch_size=100, device=torch.device('cpu'))
    print(inputs.shape, targets.shape)

