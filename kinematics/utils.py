import numpy as np


# Create a rotation and translation matrix based on DH parameters
def create_dh_matrix(theta, alpha, a, d, radians=True):
    if not radians:
        alpha = np.radians(alpha)
        theta = np.radians(theta)

    A = np.array(
        [[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
         [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
         [0, np.sin(alpha), np.cos(alpha), d],
         [0, 0, 0, 1]])

    return A


# Create a Jacobi Matrix based on DH parameters for inverse kinematic through gradient descent
def create_jacobian(thetas,
                    a_sh,
                    a_el,
                    arm='right',
                    radians=True):

    if not radians:
        thetas = np.radians(thetas)

    if isinstance(thetas, tuple | list):
        thetas = np.array(thetas)

    theta_sh, theta_el = thetas

    if arm == 'right':
        const = 1
    elif arm == 'left':
        theta_sh = np.pi - theta_sh
        theta_el = - theta_el
        const = - 1
    else:
        raise ValueError('Please specify if the arm is right one or left one!')

    # Calculate the derivatives of the end effector position with respect to the joint angles
    J11 = - a_sh * np.sin(theta_sh) - a_el * np.sin(theta_sh + theta_el)
    J12 = - a_el * np.sin(theta_sh + theta_el)
    J21 = a_sh * np.cos(theta_sh) + a_el * np.cos(theta_sh + theta_el)
    J22 = a_el * np.cos(theta_sh + theta_el)

    # Assemble the Jacobian matrix
    J = const * np.array([[J11, J12], [J21, J22]])

    return J
