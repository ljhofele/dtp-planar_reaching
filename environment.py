import numpy as np
from typing import Tuple, Optional
from kinematics.planar_arms import PlanarArms


def generate_random_coordinate(arm: str,
                               theta_bounds_lower: np.ndarray,
                               theta_bounds_upper: np.ndarray,
                               x_bounds: Tuple[float, float],
                               y_bounds: Tuple[float, float],
                               clip_borders_xy: float = 10.,
                               min_distance: float = 50.,
                               init_thetas: Optional[np.ndarray] = None,
                               return_thetas_radians: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    valid = False
    while not valid:
        random_thetas = np.random.uniform(low=theta_bounds_lower,
                                          high=theta_bounds_upper,
                                          size=2)

        random_xy = PlanarArms.forward_kinematics(arm=arm,
                                                  thetas=random_thetas,
                                                  radians=False)[:, -1]

        if (x_bounds[0] + clip_borders_xy < random_xy[0] < x_bounds[1] - clip_borders_xy
                and y_bounds[0] + clip_borders_xy < random_xy[1] < y_bounds[1] - clip_borders_xy):
            # check if thetas are far from each other
            if init_thetas is not None:
                # init thetas must be in degrees
                init_xy = PlanarArms.forward_kinematics(arm=arm,
                                                        thetas=init_thetas,
                                                        radians=False)[:, -1]
                if np.linalg.norm(init_xy - random_xy) > min_distance:
                    valid = True
            else:
                valid = True

    if return_thetas_radians:
        random_thetas = np.radians(random_thetas)

    return random_thetas, random_xy


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
