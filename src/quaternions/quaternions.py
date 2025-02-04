from warnings import warn

import numpy as np


def _validate_input(input_data, expected_shape):
    """
    Validate the input data type and shape.

    Parameters
    ----------
    input_data : np.ndarray
        Input data to be validated.
    expected_shape : str
        Expected data shape ("quaternion", "rpy", or "rotmat").

    Raises
    ------
    TypeError
        If the input data type does not match the expected data type.
    ValueError
        If the input data shape does not match the expected shape.
    """
    if not isinstance(input_data, np.ndarray):
        raise TypeError(f"Expected numpy array but got {type(input_data)}")

    shape_map = {"quaternion": (4,), "rpy": (3,), "rotmat": (3, 3)}

    if expected_shape not in shape_map:
        raise ValueError(f"Unknown expected shape: {expected_shape}")

    expected_shape_shape = shape_map[expected_shape]
    if input_data.ndim == 1:
        if input_data.shape != expected_shape_shape:
            raise ValueError(f"Expected shape {expected_shape_shape} but got {input_data.shape}")
    elif input_data.ndim == 2 and expected_shape != "rotmat":
        if input_data.shape[1:] != expected_shape_shape:
            raise ValueError(f"Expected shape (n, {expected_shape_shape}) but got {input_data.shape}")
    elif input_data.ndim == 2 and expected_shape == "rotmat":
        if input_data.shape != expected_shape_shape:
            raise ValueError(f"Expected shape ({expected_shape_shape}) but got {input_data.shape}")
    elif input_data.ndim == 3 and expected_shape == "rotmat":
        if input_data.shape[1:] != expected_shape_shape:
            raise ValueError(f"Expected shape (n, {expected_shape_shape}) but got {input_data.shape}")
    else:
        raise ValueError(f"Invalid input data dimensions: {input_data.ndim}")


def to_scalar_first(q):
    """
    Converts a quaternion from scalar last to scalar first.

    Parameters
    ----------
    q : ndarray
        Quaternion with scalar last.

    Returns
    -------
    ndarray
        Quaternion with scalar first.
    """
    if (q.ndim > 1) and (np.argmax(q.shape) == 0):
        q = q.T
        return q[[3, 0, 1, 2], :]
    else:
        return q[[3, 0, 1, 2]]


def conjugate(q, scalarLast=False):
    """
    Returns the conjugate of a quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion assuming scalar first.

    Returns
    -------
    ndarray
        Conjugate of q.
    """
    q_out = q * np.array([1, -1, -1, -1]) if not scalarLast else q * np.array([-1, -1, -1, 1])
    return q_out


def inverse(q, scalarLast=False):
    """
    Returns the inverse of a quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion assuming scalar first.

    Returns
    -------
    ndarray
        Inverse of q.
    """

    return (conjugate(q, scalarLast).T / np.linalg.norm(q, axis=1 if q.ndim > 1 else None).T ** 2).T


def exponential(q, scalarLast=False):
    """
    Returns the exponential of a quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion assuming scalar first.

    Returns
    -------
    ndarray
        Exponential of q.
    """
    if is_real(q):
        return q
    q0 = q[0] if not scalarLast else q[3]
    qv = q[1:4] if not scalarLast else q[0:3]
    qv_norm = np.linalg.norm(qv)
    eq0 = (np.exp(q0) * np.cos(qv_norm),)
    eqv = np.exp(q0) * (qv / qv_norm) * np.sin(qv_norm)
    return np.concatenate([eq0, eqv]) if not scalarLast else np.concatenate([eqv, eq0])


def logarithm(q, scalarLast=False):
    """
    Returns the logarithm of a quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion.

    Returns
    -------
    ndarray
        Logarithm of q.
    """
    q0 = q[0] if not scalarLast else q[3]
    qv = q[1:4] if not scalarLast else q[0:3]
    qv_norm = np.linalg.norm(qv)

    if qv_norm == 0.0:
        return np.zeros(4)

    logq0 = (np.log(np.linalg.norm(q)),)
    logqv = qv * np.arccos(q0 / np.linalg.norm(q)) / qv_norm

    return np.concatenate([logq0, logqv]) if not scalarLast else np.concatenate([logqv, logq0])


def normalize(q, scalarLast=False):
    """
    Returns the normalized quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion assuming scalar first.

    Returns
    -------
    ndarray
        Normalized q.

    Notes
    -----
    This function calculates the normalized quaternion.

    """
    _validate_input(q, "quaternion")
    if q.ndim > 1:
        if any(np.linalg.norm(q, axis=1 if q.ndim > 1 else None) == 0.0):
            raise ArithmeticError("Zero division error, there are quaternions with zero norm.")
    elif np.linalg.norm(q, axis=1 if q.ndim > 1 else None) == 0.0:
        raise ArithmeticError("Zero division error, there are quaternions with zero norm.")

    return (q.T / np.linalg.norm(q, axis=1 if q.ndim > 1 else None).T).T


def product(q, p, scalarLast=False):
    """
    Returns the Hamilton product of two quaternions.

    Parameters
    ----------
    q : ndarray
        Quaternion(s) in scalar-first or scalar-last format. Shape can be (4,) or (n, 4).
    p : ndarray
        Quaternion(s) in scalar-first or scalar-last format. Shape can be (4,) or (n, 4).
    scalarLast : bool, optional
        Whether the input quaternions are in scalar-last format (default is False).

    Returns
    -------
    ndarray
        Hamilton product of q and p, with the same shape as the inputs.
    """
    # Ensure inputs are valid quaternions
    _validate_input(q, "quaternion")
    _validate_input(p, "quaternion")

    # Adjust dimensions for batch processing
    q = np.atleast_2d(q)
    p = np.atleast_2d(p)

    # Convert to scalar-first format if necessary
    if scalarLast:
        q = to_scalar_first(q)
        p = to_scalar_first(p)

    # Extract components of the quaternions
    q0, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    p0, px, py, pz = p[:, 0], p[:, 1], p[:, 2], p[:, 3]

    # Compute the Hamilton product
    prod = np.column_stack(
        (
            q0 * p0 - qx * px - qy * py - qz * pz,
            q0 * px + qx * p0 + qy * pz - qz * py,
            q0 * py - qx * pz + qy * p0 + qz * px,
            q0 * pz + qx * py - qy * px + qz * p0,
        )
    )

    # Return shape consistent with input
    return prod if len(prod) > 1 else prod[0]


def is_real(q):
    """
    Returns True if the quaternion is real (i.e., vector part is zero).

    Parameters
    ----------
    q : ndarray
        Quaternion(s) in scalar-first or scalar-last format. Shape can be (4,) or (n, 4).

    Returns
    -------
    ndarray
        Boolean array indicating if the quaternion is real.
    """
    # Ensure input is a valid quaternion
    _validate_input(q, "quaternion")

    # Check if the vector part is zero
    return np.all(q[:, 1:] == 0, axis=1) if q.ndim > 1 else np.all(q[1:] == 0)


def is_pure(q):
    """
    Returns True if the quaternion is pure (i.e., scalar part is zero).

    Parameters
    ----------
    q : ndarray
        Quaternion(s) in scalar-first or scalar-last format. Shape can be (4,) or (n, 4).

    Returns
    -------
    ndarray
        Boolean array indicating if the quaternion is pure.
    """
    # Ensure input is a valid quaternion
    _validate_input(q, "quaternion")

    # Check if the scalar part is zero
    return (q[:, 0] == 0) if q.ndim > 1 else (q[0] == 0)


def is_unit(q):
    """
    Returns True if the quaternion is a unit quaternion.

    Parameters
    ----------
    q : ndarray
        Quaternion(s) in scalar-first or scalar-last format. Shape can be (4,) or (n, 4).

    Returns
    -------
    ndarray
        Boolean array indicating if the quaternion is a unit quaternion.
    """
    # Ensure input is a valid quaternion
    _validate_input(q, "quaternion")

    # Check if the quaternion is normalized
    return np.allclose(np.linalg.norm(q, axis=1 if q.ndim > 1 else None), 1.0)


def identity(n=1):
    """
    Returns the identity quaternion.

    Parameters
    ----------
    n : int, optional
        Number of identity quaternions to generate (default is 1).

    Returns
    -------
    ndarray
        Identity quaternion(s) with shape (n, 4).
    """
    return np.hstack((np.ones((n, 1)), np.zeros((n, 3))))


def quat_rotate(q, v):
    """
    Rotate a vector or batch of vectors using a quaternion or batch of quaternions.

    This function applies the quaternion rotation formula to rotate 3D vectors `v` by quaternions `q`.
    The rotation is performed using the operation:
        q * v * q^-1
    where `v` is treated as a pure quaternion with a scalar part of zero.

    Parameters:
    ----------
    q : array-like
        A 4-element array for a single quaternion or an (N, 4) array for a batch of quaternions,
        where each quaternion is in the format [q1, q2, q3, q4]:
        - q1: scalar part of the quaternion.
        - q2, q3, q4: vector (imaginary) components of the quaternion.
    v : array-like
        A 3-element array for a single vector or an (N, 3) array for a batch of vectors to be rotated.

    Returns:
    -------
    numpy.ndarray
        A 3-element array for a single rotated vector or an (N, 3) array for a batch of rotated vectors.

    Notes:
    -----
    - The quaternion `q` should be normalized for correct rotation.
    - The scalar part of the resulting quaternion is discarded, returning only the vector part.
    """
    q = np.asarray(q)
    v = np.asarray(v)

    # Ensure input shapes are correct for batch processing
    if q.ndim == 1:  # Single quaternion
        q = q[np.newaxis, :]  # Add batch dimension (1, 4)
    if v.ndim == 1:  # Single vector
        v = v[np.newaxis, :]  # Add batch dimension (1, 3)

    if q.shape[0] != v.shape[0]:
        raise ValueError(f"The number of quaternions and vectors must match in batch size. Quaternion batch size: {q.shape[0]}, Vector batch size: {v.shape[0]}")

    # Convert vectors to quaternions with scalar part 0
    v_quat = np.hstack((np.zeros((v.shape[0], 1)), v))  # Shape: (N, 4)
    # Perform quaternion rotation: rotated_quat = q * v_quat * conjugate(q)
    q_conj = conjugate(q)  # Compute conjugates for all quaternions in the batch
    rotated_quat = product(product(q, v_quat), q_conj)  # Batch quaternion multiplication

    # Extract vector part of the result (ignore scalar part)
    if rotated_quat.ndim == 1:  # Single vector case
        return rotated_quat[1:]  # Return a 1D array (3,)
    else:  # Batch case
        return rotated_quat[:, 1:]  # Return a 2D array (N, 3)


def yaw_quat(q):
    """
    Calculate the yaw (rotation about the vertical axis) from a quaternion or batch of quaternions.

    The quaternion is expected in the form [q1, q2, q3, q4], where:
    - q1: scalar part of the quaternion.
    - q2, q3, q4: vector (imaginary) components of the quaternion.

    The yaw is computed based on the rotation matrix representation derived from the quaternion.
    The result is expressed in degrees.

    Parameters:
    ----------
    q : array-like
        A 4-element array or an (N, 4) array for a batch of quaternions, each in the format [q1, q2, q3, q4].

    Returns:
    -------
    numpy.ndarray or float
        The yaw angle(s) in degrees, ranging from -180 to 180.

    Notes:
    -----
    - Positive yaw corresponds to a counterclockwise rotation about the vertical axis (Z-axis).
    - This calculation assumes a right-handed coordinate system.
    """
    q = np.asarray(q)

    # Ensure q has the correct shape
    if q.ndim == 1:  # Single quaternion
        q = q[np.newaxis, :]  # Add batch dimension (1, 4)

    # Extract quaternion components
    q1, q2, q3, q4 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Calculate yaw for each quaternion in the batch
    a12 = 2 * (q2 * q3 + q1 * q4)
    a22 = q1 * q1 + q2 * q2 - q3 * q3 - q4 * q4

    yaw = np.degrees(np.arctan2(a12, a22))  # Compute yaw angles in degrees

    if yaw.shape[0] == 1:  # Single quaternion case
        return yaw[0]  # Return a single value (float)
    else:  # Batch case
        return yaw  # Return an array of yaw values for each quaternion


def to_angles(q, scalarLast=False):
    """
     Returns the angles of a quaternion.

     Parameters
     ----------
     q : ndarray
         Quaternion assuming scalar first.
     scalarLast : bool, optional
         Flag indicating whether the scalar component is the last element of the quaternion.
         Default is False.

     Returns
     -------
     ndarray
         Angles of the quaternion in phi, theta, psi.

     Notes
     -----
     This function calculates the angles of a quaternion using the formula from the paper
     "Quaternion to Euler Angle Conversion for Arbitrary Rotation Sequence" by Shuster and Oh.
     The implementation is based on the pytransform3d and scipy libraries for XYZ sequence.

    References
     ----------
     - Bernardes & Viollet (2018). "Quaternion to Euler Angle Conversion for Arbitrary Rotation Sequence". Journal of Aerospace Technology and Management. https://doi.org/10.1371/journal.pone.0276302
     - NASA Mission Planning and Analysis Division (July 1977). "Euler Angles, Quaternions, and Transformation Matrices". NASA
     - scipy: https://www.scipy.org/

    """
    if q.ndim > 1:
        q = q.T

    if scalarLast:
        q0, qx, qy, qz = to_scalar_first(q)
    else:
        q0, qx, qy, qz = q.squeeze()

    phi = np.arctan2(qx + qz, q0 - qy) - np.arctan2(qz - qx, qy + q0)
    theta = np.arccos((q0 - qy) ** 2 + (qx + qz) ** 2 - 1) - np.pi / 2
    psi = np.arctan2(qx + qz, q0 - qy) + np.arctan2(qz - qx, qy + q0)

    euler_angles = np.array([phi, theta, psi])
    # validate data dimensions in correct order
    if np.argmax(euler_angles.shape) != 0:
        euler_angles = euler_angles.T
    return euler_angles


def from_angles(angles: np.ndarray, order="rpy"):
    """
    Returns the quaternion from a series of roll (about x axis), pitch (about y axis), and yaw (about z axis) angles.

    Parameters
    ----------
    angles : ndarray
        Roll, pitch, yaw angles. Can be 1D (e.g., [r, p, y]) or 2D with nx3 (e.g., [[r_t1, p_t1, y_t1], [r_t2, p_t2, y_t2], ...]).
    order : str, optional
        Order of the angles. Default is "rpy" or "XYZ".
    Returns
    -------
    ndarray
        Quaternion in scalar first form.

    References
    ----------
     - Bernardes & Viollet (2018). "Quaternion to Euler Angle Conversion for Arbitrary Rotation Sequence". Journal of Aerospace Technology and Management. https://doi.org/10.1371/journal.pone.0276302
     - NASA Mission Planning and Analysis Division (July 1977). "Euler Angles, Quaternions, and Transformation Matrices". NASA
     - scipy: https://www.scipy.org/
    """
    if angles.ndim > 1:
        angles = angles.T
    roll, pitch, yaw = angles

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    if order == "rpy":
        q = np.array(
            [
                -sr * sp * sy + cr * cp * cy,
                sr * cp * cy + sp * sy * cr,
                -sr * sy * cp + sp * cr * cy,
                sr * sp * cy + sy * cr * cr,
            ]
        )
    elif order == "ryp":
        q = np.array(
            [
                sr * sy * sp + cr * cy * cp,
                sr * cy * cp - sy * sp * cr,
                -sr * sy * cp + sp * cr * cy,
                sr * sp * cy + sy * cr * cr,
            ]
        )

    # validate data dimensions in correct order
    if np.argmax(q.shape) != 0:
        q = q.T
    return q


def to_rotmat(q, scalarLast=False, homogenous=True):
    """Converts a quaternion to a right hand rotation matrix. Choice of inhomogeneous or homogeneous representation (latter is preferred). If the quaternion is not a unit quaternion, the homogenous representation is still a scalar multiple of a rotation matrix while the inhomogeneous representation is not an orthogonal rotation matrix.

    Parameters
    ----------
    q : ndarray
        unit quaternion in shape (4,) or (n, 4)
    scalarLast : bool, optional
        Flag indicating whether the quaternion is in scalar-last format. Default is False
    homogenous : bool, optional
        Flag indicating whether the rotation matrix should be in homogeneous form. Default is True

    Returns
    -------
    ndarray
        Rotation matrix representation of the original quaternion.
    """
    q = normalize(q)
    if scalarLast:
        q = to_scalar_first(q)
    if q.ndim == 1:
        q = q[np.newaxis, :]  # Add batch dimension if single quaternion

    R = np.full((q.shape[0], 3, 3), np.nan)
    for i in range(q.shape[0]):
        q0, qx, qy, qz = q[i, 0], q[i, 1], q[i, 2], q[i, 3]
        if homogenous:
            R[i, :, :] = np.array(
                [
                    [
                        q0**2 + qx**2 - qy**2 - qz**2,
                        2 * (qx * qy - q0 * qz),
                        2 * (q0 * qy + qx * qz),
                    ],
                    [
                        2 * (qx * qy + q0 * qz),
                        q0**2 - qx**2 + qy**2 - qz**2,
                        2 * (qy * qz - q0 * qx),
                    ],
                    [
                        2 * (qx * qz - q0 * qy),
                        2 * (q0 * qx + qy * qz),
                        q0**2 - qx**2 - qy**2 + qz**2,
                    ],
                ]
            )
        else:
            R[i, :, :] = np.array(
                [
                    [
                        1.0 - 2.0 * (qy**2 + qz**2),
                        2.0 * (qx * qy - q0 * qz),
                        2.0 * (qx * qz + q0 * qy),
                    ],
                    [
                        2.0 * (qx * qy + q0 * qz),
                        1.0 - 2.0 * (qx**2 + qz**2),
                        2.0 * (qy * qz - q0 * qx),
                    ],
                    [
                        2.0 * (qx * qz - q0 * qy),
                        2.0 * (qy * qz + q0 * qx),
                        1.0 - 2.0 * (qx**2 + qy**2),
                    ],
                ]
            )
    return R.squeeze() if R.shape[0] == 1 else R
    # return R_out if R_out.shape[0] > 1 else R_out.squeeze()


def from_rotmat(R: np.ndarray):
    """
    Converts a 3x3 orthonormal rotation matrix to a quaternion in scalar first form.

    Parameters
    ----------
    R : np.ndarray
        Orthogonal rotation matrix 3x3 or Nx3x3. where N is time samples of each 3x3 matrix

    Returns
    -------
    np.ndarray
        Quaternion in scalar first form.

    Raises
    ------
    ValueError
        If R is not a 2 or 3 dimensional matrix or if R shape is not (3, 3).

    Warns
    -----
    UserWarning
        If R is not orthogonal.

    Notes
    -----
    This function expects a 3x3 or Nx3x3 matrix. If you pass a matrix with a different shape, a ValueError will be raised.
    If R is not orthogonal, a UserWarning will be issued.

    Somewhat slow running due to the for loops need in the event of passing an Nx3x3 matrix. Not sure how to vectorize this...
    """
    # largest_dim = np.argmax(R.shape)
    if R.ndim not in [2, 3]:
        raise ValueError("R must be a 2 or 3 dimensional matrix")
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Function expects a 3x3 or Nx3x3 matrix. You passed a {R.shape} matrix.")
    if R.ndim == 3:
        for i in range(R.shape[0]):
            if not np.allclose(np.dot(R[i, :, :], R[i, :, :].T), np.eye(3), atol=1e-6):
                warn("R is not orthogonal")
    else:
        if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6):
            warn("R is not orthogonal")
    q_out = []
    if R.ndim == 2:
        R = R[np.newaxis, :, :]
    for i in range(R.shape[0]):
        r = R[i, :, :]
        q = np.empty((4))
        trace = np.trace(r)

        if trace > 0.0:
            sqrt_trace = np.sqrt(1.0 + trace)
            q[0] = 0.5 * sqrt_trace
            q[1] = 0.5 / sqrt_trace * (r[2, 1] - r[1, 2])
            q[2] = 0.5 / sqrt_trace * (r[0, 2] - r[2, 0])
            q[3] = 0.5 / sqrt_trace * (r[1, 0] - r[0, 1])
        else:
            if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
                sqrt_trace = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
                q[0] = 0.5 / sqrt_trace * (r[2, 1] - r[1, 2])
                q[1] = 0.5 * sqrt_trace
                q[2] = 0.5 / sqrt_trace * (r[1, 0] + r[0, 1])
                q[3] = 0.5 / sqrt_trace * (r[0, 2] + r[2, 0])
            elif r[1, 1] > r[2, 2]:
                sqrt_trace = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
                q[0] = 0.5 / sqrt_trace * (r[0, 2] - r[2, 0])
                q[1] = 0.5 / sqrt_trace * (r[1, 0] + r[0, 1])
                q[2] = 0.5 * sqrt_trace
                q[3] = 0.5 / sqrt_trace * (r[2, 1] + r[1, 2])
            else:
                sqrt_trace = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
                q[0] = 0.5 / sqrt_trace * (r[1, 0] - r[0, 1])
                q[1] = 0.5 / sqrt_trace * (r[0, 2] + r[2, 0])
                q[2] = 0.5 / sqrt_trace * (r[2, 1] + r[1, 2])
                q[3] = 0.5 * sqrt_trace

        q_out.append(q / np.linalg.norm(q, axis=0, keepdims=True))
    q_out_arr = np.vstack(q_out)
    return q_out_arr if q_out_arr.shape[0] > 1 else q_out_arr.squeeze()


def from_axis_angle(ax: np.ndarray, angleFirst=True):
    """
    Convert a rotation in axis angle form to a quaternion in scalar first form.

    Parameters:
        ax (np.ndarray): Array containing the unit vectors of the axis and the angle.
        angleFirst (bool, optional): Order of elements in ax. Defaults to False.

    Returns:
        np.ndarray: Quaternion in scalar first form.
    """

    if ax.ndim == 1:
        ax = ax[np.newaxis, :]

    if angleFirst:
        angle = ax[:, 0]
        axis = ax[:, 1:]
    else:
        angle = ax[:, -1]
        axis = ax[:, :-1]

    q = np.empty((ax.shape[0], 4))
    for i in range(ax.shape[0]):
        if all(ax[i, :] == 0):
            q[i, :] = np.array([1, 0, 0, 0])
        q[i, 0] = np.cos(angle[i] / 2)
        q[i, 1:] = np.sin(angle[i] / 2) * axis[i] / np.linalg.norm(axis[i])
    return q
