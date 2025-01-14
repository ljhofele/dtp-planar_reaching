# Planar Reaching with Difference Target Propagation

This repository implements a learning framework for planar reaching movements using Difference Target Propagation (DTP). The implementation includes both forward kinematics for a 2-joint planar arm and a DTP network for learning reaching movements.

## Overview

The project consists of two main components:

1. **Planar Arm Kinematics**: A 2-joint robotic arm model with:
   - Forward kinematics for end-effector positioning
   - Inverse kinematics for joint angle calculation
   - Workspace constraints and joint limits
   - Visualization capabilities

2. **Difference Target Propagation Network**: A neural network that learns to generate reaching movements with:
   - Layer-wise training using DTP
   - Feedback weight learning through reconstruction
   - Forward/backward pathway optimization
   - Target propagation for hidden layers

## Project Structure

```
esyniawa-dtp-planar_reaching/
├── LICENSE                 # MIT License
├── environment.py         # Environment setup and data handling
├── main.py               # Training script and network creation (example)
├── kinematics/          
│   ├── planar_arms.py    # Planar arm implementation
│   └── utils.py          # Kinematics utilities
└── network/
    └── dtp.py            # DTP network implementation
```

## Implementation Details

### Planar Arm
- Two-joint arm with shoulder and elbow
- Configurable link lengths and joint limits
- Support for both left and right arm configurations
- Smooth trajectory generation with cosine interpolation

### DTP Network
- Customizable layer architecture
- Forward and feedback weight optimization
- Local target computation for each layer
- MSE-based reconstruction loss
- Built-in support for batch processing

## Dependencies

- PyTorch
- NumPy
- Matplotlib

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Based on research in target propagation and biological learning mechanisms, particularly:
- Lee et al. (2015) "Difference Target Propagation"
- Meulemans et al. (2020) "A Theoretical Framework for Target Propagation"
