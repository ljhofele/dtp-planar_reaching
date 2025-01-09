import numpy as np
import torch
from typing import Tuple, Optional

from torch import Tensor

from kinematics.planar_arms import PlanarArms


def data_transform(thetas: np.ndarray,
                   xy: np.ndarray, ):
    """
    Transforms angles in radians to sin and cos values and concatenates with xy coordinates
    :param thetas: angle in radians
    :param xy: cartesian coordinates
    :return: inputs and targets
    """
    return np.concatenate((np.sin(thetas), np.cos(thetas), xy), axis=1)


def create_batch(
        arm: str,
        batch_size: int,
        device: torch.device) -> Tuple[Tensor, Tensor]:
    init_shoulder_theta = np.random.uniform(low=PlanarArms.l_upper_arm_limit, high=PlanarArms.u_upper_arm_limit)
    init_elbow_theta = np.random.uniform(low=PlanarArms.l_forearm_limit, high=PlanarArms.u_forearm_limit)

    init_thetas = np.array((init_shoulder_theta, init_elbow_theta))

    delta_thatas, targets_xy = generate_random_coordinates(arm=arm,
                                                           initial_thetas=init_thetas,
                                                           num_movements=batch_size,
                                                           return_thetas_radians=False)
    # input for network
    init_thetas = np.repeat(init_thetas, batch_size).reshape((batch_size, -1))
    input_batch = data_transform(thetas=init_thetas,
                                 xy=targets_xy)

    # target for network
    target_batch = data_transform(thetas=delta_thatas,
                                  xy=targets_xy)

    return (torch.from_numpy(input_batch).float().to(device),  # input (batch_size, 6)
            torch.from_numpy(target_batch).float().to(device))  # target (batch_size, 6)


def generate_random_coordinates(arm: str,
                                initial_thetas: np.ndarray,  # in radians
                                num_movements: int,
                                x_bounds: Tuple[float, float] = PlanarArms.x_limits,
                                y_bounds: Tuple[float, float] = PlanarArms.y_limits,
                                min_distance: float = 50.,
                                return_thetas_radians: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random reaching points that are at least min_distance away from initial position.

    Args:
        arm: Type of planar arm
        initial_thetas: Initial joint angles in radians
        num_movements: Number of random points to generate
        x_bounds: X-axis bounds for the workspace
        y_bounds: Y-axis bounds for the workspace
        min_distance: Minimum distance required from initial position
        return_thetas_radians: Whether to return angles in radians

    Returns:
        Tuple of (joint angle changes, normalized xy coordinates)
    """
    lower_arm_limits, upper_arm_limits = PlanarArms.get_bounds()
    init_pos = PlanarArms.forward_kinematics(arm=arm,
                                             thetas=initial_thetas,
                                             radians=False)[:, -1]

    random_movement_thetas = []
    random_xy = []

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
            normalized_xy = norm_xy(current_pos, x_bounds=x_bounds, y_bounds=y_bounds)

            random_movement_thetas.append(theta_change)
            random_xy.append(normalized_xy)

    return np.array(random_movement_thetas), np.array(random_xy)


def norm_xy(xy: np.ndarray,
            x_bounds: Tuple[float, float],
            y_bounds: Tuple[float, float],
            clip_borders_xy: float = 10.) -> np.ndarray:
    x_bounds = (x_bounds[0] + clip_borders_xy, x_bounds[1] - clip_borders_xy)
    y_bounds = (y_bounds[0] + clip_borders_xy, y_bounds[1] - clip_borders_xy)

    # Calculate the midpoints of x and y ranges
    x_mid = (x_bounds[0] + x_bounds[1]) / 2
    y_mid = (y_bounds[0] + y_bounds[1]) / 2

    # Calculate the half-ranges
    x_half_range = (x_bounds[1] - x_bounds[0]) / 2
    y_half_range = (y_bounds[1] - y_bounds[0]) / 2

    # Normalize to [-1, 1]
    normalized_x = (xy[0] - x_mid) / x_half_range
    normalized_y = (xy[1] - y_mid) / y_half_range

    return np.array([normalized_x, normalized_y])


if __name__ == "__main__":
    thetas, xys = generate_random_coordinates(arm='right',
                                              initial_thetas=np.array([0., 0.]),
                                              num_movements=100, )

    print(thetas.shape, xys.shape)

    new_inputs, new_targets = create_batch(arm='right', batch_size=10, device=torch.device("cpu"))
    print(new_inputs.shape, new_targets.shape)
