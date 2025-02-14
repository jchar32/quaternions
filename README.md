# Python Quaternions
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/jchar32/quaternions/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/jchar32/quaternions?branch=main)

A lightweight Python package for handling quaternions. This package provides efficient tools for working with quaternions, including conversions between different rotation representations (Euler angles, rotation matrices) and common quaternion operations.


## Quick Examples

### 1. Basic Quaternion Operations

```bash
pip install https://github.com/jchar32/quaternions.git
```

Verify installation:
```python
import quaternions
print(quaternions.__version__)
```


```python
import numpy as np
from quaternions import normalize, product

# Create two quaternions (scalar-first format)
q1 = np.array([1, 0, 1, 0])  # [w, x, y, z]
q2 = np.array([0, 1, 0, 1])

# Normalize and multiply quaternions
q1_norm = normalize(q1)
q2_norm = normalize(q2)
result = product(q1_norm, q2_norm)
```

### 2. Rotating Vectors
```python
from quaternions import from_angles, quat_rotate

# Create rotation quaternion from Euler angles (in radians)
angles = np.array([np.pi/4, 0, 0])  # 45° rotation around x-axis
q = from_angles(angles)

# Rotate a vector
vector = np.array([0, 0, 1])
rotated_vector = quat_rotate(q, vector)
```

### 3. Converting Between Representations
```python
from quaternions import from_rotmat, to_angles

# Create rotation matrix
R = np.array([[0, -1, 0],
              [1,  0, 0],
              [0,  0, 1]])

# Convert to quaternion and then to Euler angles
q = from_rotmat(R)
euler_angles = to_angles(q, order='xyz')
print(f"Euler angles (degrees): {np.degrees(euler_angles)}")
```

For more detailed documentation and examples, visit our [documentation page](https://quaternions.readthedocs.io/).