import numpy as np


def _validate_input(input_data, expected_shape):
    """
    Validate the input data type and shape.

    Parameters
    ----------
    input_data : np.ndarray
        Input data to be validated.
    expected_shape : str
        Expected data shape ("quaternion", "euler", or "rotmat").

    Raises
    ------
    TypeError
        If the input data type does not match the expected data type.
    ValueError
        If the input data shape does not match the expected shape or contains nonfinite elements.
    """
    if not isinstance(input_data, np.ndarray):
        raise TypeError(f"Expected numpy array but got {type(input_data)}")

    if not np.all(np.isfinite(input_data)):
        raise ValueError("Input data contains non-finite values")

    shape_map = {"quaternion": (4,), "euler": (3,), "rotmat": (3, 3)}

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
    Converts a quaternion from scalar-last to scalar-first format.

    Parameters
    ----------
    q : np.ndarray
        Quaternion with scalar last.

    Returns
    -------
    np.ndarray
        Quaternion with scalar first.
    """
    if (q.ndim > 1) and (np.argmax(q.shape) == 0):
        q = q.T
        return q[[3, 0, 1, 2], :].T
    elif q.shape == (1, 4):
        return q[:, [3, 0, 1, 2]]
    else:
        return q[[3, 0, 1, 2]]


def conjugate(q, scalarLast=False):
    """
    Returns the conjugate of a quaternion.

    Parameters
    ----------
    q : np.ndarray
        Quaternion, assumed to be in scalar-first format.
    scalarLast : bool, optional
        If True, q is assumed to be in scalar-last format. Default is False.

    Returns
    -------
    np.ndarray
        Conjugate of q.
    """
    q_out = q * np.array([1, -1, -1, -1]) if not scalarLast else q * np.array([-1, -1, -1, 1])
    return q_out


def inverse(q, scalarLast=False):
    """
    Returns the inverse of a quaternion.

    Parameters
    ----------
    q : np.ndarray
        Quaternion, assumed to be in scalar-first format.
    scalarLast : bool, optional
        If True, q is assumed to be in scalar-last format. Default is False.

    Returns
    -------
    np.ndarray
        Inverse of q.
    """

    return (conjugate(q, scalarLast).T / np.linalg.norm(q, axis=1 if q.ndim > 1 else None).T ** 2).T


def exponential(q, scalarLast=False):
    """
    Returns the exponential of a quaternion.

    Parameters
    ----------
    q : np.ndarray
        Quaternion, assumed to be in scalar-first format.
    scalarLast : bool, optional
        If True, q is assumed to be in scalar-last format. Default is False.

    Returns
    -------
    np.ndarray
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
    q : np.ndarray
        Quaternion.
    scalarLast : bool, optional
        If True, q is assumed to be in scalar-last format. Default is False.

    Returns
    -------
    np.ndarray
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
    q : np.ndarray
        Quaternion, assumed to be in scalar-first format.
    scalarLast : bool, optional
        If True, q is assumed to be in scalar-last format. Default is False.

    Returns
    -------
    np.ndarray
        Normalized quaternion.

    Raises
    ------
    ArithmeticError
        If there is a division by zero when normalizing.
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
    q : np.ndarray
        Quaternion(s) in scalar-first or scalar-last format (shape (4,) or (n, 4)).
    p : np.ndarray
        Quaternion(s) in scalar-first or scalar-last format (shape (4,) or (n, 4)).
    scalarLast : bool, optional
        If True, input quaternions are in scalar-last format. Default is False.

    Returns
    -------
    np.ndarray
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
    Returns True if the quaternion is real (i.e., its vector part is zero).

    Parameters
    ----------
    q : np.ndarray
        Quaternion(s) in scalar-first or scalar-last format.

    Returns
    -------
    bool or np.ndarray
        Boolean value (or array) indicating if the quaternion is real.
    """
    # Ensure input is a valid quaternion
    _validate_input(q, "quaternion")

    # Check if the vector part is zero
    return np.all(q[:, 1:] == 0, axis=1) if q.ndim > 1 else np.all(q[1:] == 0)


def is_pure(q):
    """
    Returns True if the quaternion is pure (i.e., its scalar part is zero).

    Parameters
    ----------
    q : np.ndarray
        Quaternion(s) in scalar-first or scalar-last format.

    Returns
    -------
    bool or np.ndarray
        Boolean value (or array) indicating if the quaternion is pure.
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
    q : np.ndarray
        Quaternion(s) in scalar-first or scalar-last format.

    Returns
    -------
    bool or np.ndarray
        Boolean value (or array) indicating if the quaternion is a unit quaternion.
    """
    # Ensure input is a valid quaternion
    _validate_input(q, "quaternion")

    # Check if the quaternion is normalized
    return np.allclose(np.linalg.norm(q, axis=1 if q.ndim > 1 else None), 1.0)


def identity(n=1):
    """
    Returns the identity quaternion(s).

    Parameters
    ----------
    n : int, optional
        Number of identity quaternions to generate. Default is 1.

    Returns
    -------
    np.ndarray
        Identity quaternion(s) with shape (n, 4).
    """
    return np.hstack((np.ones((n, 1)), np.zeros((n, 3))))


def quat_rotate(q, v):
    """
    Rotate a vector or a batch of vectors using a quaternion or batch of quaternions.

    Parameters
    ----------
    q : array-like
        A 4-element array for a single quaternion or an (N, 4) array for a batch of quaternions.
    v : array-like
        A 3-element array for a single vector or an (N, 3) array for a batch of vectors.

    Returns
    -------
    np.ndarray
        A 3-element array for a rotated vector or an (N, 3) array for a batch of rotated vectors.

    Notes
    -----
    The function applies the rotation formula: q * v * q⁻¹, treating v as a pure quaternion.
    """
    _validate_input(q, "quaternion")
    _validate_input(v, "euler")

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


def yaw(q, yaw_index=2):
    """
    Extract the yaw angle from a quaternion.

    Parameters
    ----------
    q : array-like
        A 4-element array or an (N, 4) array for a batch of quaternions.

    Returns
    -------
    float or np.ndarray
        Yaw angle in degrees (or an array of yaw angles), ranging from -180 to 180.

    """
    euler_angles = np.degrees(to_angles(q, order="xyz"))
    return euler_angles[yaw_index] if euler_angles.ndim == 1 else euler_angles[:, yaw_index]


def to_angles(q: np.ndarray, order: str = "XYZ", scalarLast: bool = False):
    """
    Returns the Euler angles (phi, theta, psi) from a quaternion. Based on algorithm described at: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/Quaternions.pdf

    Parameters
    ----------
    q : np.ndarray
        Quaternion, assumed to be in scalar-first format.
    scalarLast : bool, optional
        If True, q is assumed to be in scalar-last format. Default is False.

    Returns
    -------
    np.ndarray
        Euler angles in the order (phi, theta, psi).
    """
    _validate_input(q, "quaternion")
    order = order.upper()

    if len(order) != len(set(order)):
        raise ValueError("Rotation order must not contain duplicate characters")

    q = normalize(q)

    if scalarLast:
        q = to_scalar_first(q)
    if q.ndim == 1:
        q = q[np.newaxis, :]  # Add batch dimension if single quaternion

    e = -1 if order in ["XYZ", "YZX", "ZXY"] else 1  # using same variable nameing (e) as in the reference (see docstring)

    # set the quaternion rotation order specified by order
    map = {"X": 1, "Y": 2, "Z": 3}
    q0, q1, q2, q3 = q[:, 0], q[:, map[order[0]]], q[:, map[order[1]]], q[:, map[order[2]]]

    angles = np.full((q0.shape[0], 3), 0.0)

    sin_theta = np.clip(2 * (q0 * q2 + e * q3 * q1), -1, 1)
    gimbal_lock_check = np.abs(sin_theta) == 1.0

    angles[:, 1] = np.arcsin(sin_theta)

    angles[:, 0] = np.where(gimbal_lock_check, 2 * np.arctan2(q1, q0), np.arctan2(2 * (q0 * q1 - e * q2 * q3), 1 - 2 * (q1**2 + q2**2)))

    angles[:, 2] = np.where(gimbal_lock_check, np.zeros(angles[:, 2].shape), np.arctan2(2 * (q0 * q3 - e * q1 * q2), 1 - 2 * (q2**2 + q3**2)))

    return angles.squeeze() if angles.shape[0] == 1 else angles


def from_angles(angles: np.ndarray, order: str = "XYZ"):
    """
    Returns the quaternion from a series of xyz Euler angles represented as extrinsic rotations (global frame).

    Parameters
    ----------
    angles : np.ndarray
        x, y, and z angles. Either a 1D array ([x, y, z]) or a 2D array with shape (n, 3).
    order : str, optional
        Order of the angles. Supported orders are "xyz", "xzy", "yzx", "zxy", and "zyx". Default is "xyz".

    Returns
    -------
    np.ndarray
        Quaternion in scalar-first form.

    """

    order = order.upper()
    _validate_input(angles, "euler")

    axes = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
    quaternions = [from_axis_angle([angle, *axes[axis]]) for axis, angle in zip(order, angles)]

    q = quaternions[0]
    for quat in quaternions[1:]:
        q = product(q, quat)

    return q


def to_rotmat(q: np.ndarray, scalarLast: bool = False, homogenous: bool = True):
    """
    Converts a quaternion to a right-handed rotation matrix.

    Parameters
    ----------
    q : np.ndarray
        Unit quaternion in shape (4,) or (n, 4).
    scalarLast : bool, optional
        If True, q is assumed to be in scalar-last format. Default is False.
    homogenous : bool, optional
        If True, returns a homogeneous rotation matrix. Default is True.

    Returns
    -------
    np.ndarray
        Rotation matrix representation of the quaternion.
    """
    _validate_input(q, "quaternion")
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


def from_rotmat(R: np.ndarray):
    """
    Converts a 3x3 orthonormal rotation matrix to a quaternion in scalar-first form.

    Parameters
    ----------
    R : np.ndarray
        A 3x3 or (N, 3, 3) orthogonal rotation matrix (or matrices).

    Returns
    -------
    np.ndarray
        Quaternion in scalar-first form.

    Raises
    ------
    ValueError
        If R is not a 2D or 3D matrix with shape (3, 3) or (N, 3, 3).

    """
    if R.ndim == 2:
        R = R[np.newaxis, :, :]
    _validate_input(R, "rotmat")

    if not (np.allclose(np.linalg.det(R), 1.0, atol=1e-4)):
        raise ValueError("Input matrix is not orthogonal")

    q_out = np.empty((R.shape[0], 4))
    trace = np.trace(R, axis1=1, axis2=2)
    mask = trace > 0.0
    if np.any(mask):
        sqrt_trace = np.sqrt(1.0 + trace[mask]) * 2
        q_out[mask, 0] = 0.25 * sqrt_trace
        q_out[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / sqrt_trace
        q_out[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / sqrt_trace
        q_out[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / sqrt_trace

    # trace < 0 (handle diff due to numerical instability)
    not_mask = ~mask
    if np.any(not_mask):
        diag = np.diagonal(R[not_mask], axis1=1, axis2=2)
        max_diag_idx = np.argmax(diag, axis=1)

        sqrt_trace = np.sqrt(1.0 + diag[np.arange(len(diag)), max_diag_idx]) * 2

        q_out[not_mask, 0] = (R[not_mask, 2, 1] - R[not_mask, 1, 2]) / sqrt_trace
        q_out[not_mask, 1] = (
            (max_diag_idx == 0) * (0.25 * sqrt_trace)
            + (max_diag_idx == 1) * (R[not_mask, 1, 0] + R[not_mask, 0, 1]) / sqrt_trace
            + (max_diag_idx == 2) * (R[not_mask, 0, 2] + R[not_mask, 2, 0]) / sqrt_trace
        )
        q_out[not_mask, 2] = (
            (max_diag_idx == 0) * (R[not_mask, 1, 0] + R[not_mask, 0, 1]) / sqrt_trace
            + (max_diag_idx == 1) * (0.25 * sqrt_trace)
            + (max_diag_idx == 2) * (R[not_mask, 2, 1] + R[not_mask, 1, 2]) / sqrt_trace
        )
        q_out[not_mask, 3] = (
            (max_diag_idx == 0) * (R[not_mask, 0, 2] + R[not_mask, 2, 0]) / sqrt_trace
            + (max_diag_idx == 1) * (R[not_mask, 2, 1] + R[not_mask, 1, 2]) / sqrt_trace
            + (max_diag_idx == 2) * (0.25 * sqrt_trace)
        )

    q_out /= np.linalg.norm(q_out, axis=1, keepdims=True)

    # return q_out if q_out.shape[0] > 1 else q_out.squeeze()
    # for i, idx in enumerate(max_diag_idx):
    #     r = R[not_mask][i]
    #     q = np.zeros(4)
    #     if idx == 0:
    #         sqrt_trace = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
    #         q[0] = (r[2, 1] - r[1, 2]) / (2.0 * sqrt_trace)
    #         q[1] = 0.5 * sqrt_trace
    #         q[2] = (r[1, 0] + r[0, 1]) / (2.0 * sqrt_trace)
    #         q[3] = (r[0, 2] + r[2, 0]) / (2.0 * sqrt_trace)
    #     elif idx == 1:
    #         sqrt_trace = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
    #         q[0] = (r[0, 2] - r[2, 0]) / (2.0 * sqrt_trace)
    #         q[1] = (r[1, 0] + r[0, 1]) / (2.0 * sqrt_trace)
    #         q[2] = 0.5 * sqrt_trace
    #         q[3] = (r[2, 1] + r[1, 2]) / (2.0 * sqrt_trace)
    #     else:
    #         sqrt_trace = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
    #         q[0] = (r[1, 0] - r[0, 1]) / (2.0 * sqrt_trace)
    #         q[1] = (r[0, 2] + r[2, 0]) / (2.0 * sqrt_trace)
    #         q[2] = (r[2, 1] + r[1, 2]) / (2.0 * sqrt_trace)
    #         q[3] = 0.5 * sqrt_trace

    #     q_out[not_mask][i] = q

    # q_out /= np.linalg.norm(q_out, axis=1, keepdims=True)

    return q_out if q_out.shape[0] > 1 else q_out.squeeze()


def from_axis_angle(ax: np.ndarray, angleFirst: bool = True):
    """
    Convert a rotation in axis-angle form to a quaternion in scalar-first form.

    Parameters
    ----------
    ax : np.ndarray
        Array containing the angle and the unit axis.
    angleFirst : bool, optional
        If True, the angle is the first element in `ax`. Default is True.

    Returns
    -------
    np.ndarray
        Quaternion in scalar-first form.
    """
    if not isinstance(ax, np.ndarray):
        ax = np.asarray(ax)

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
            continue
        q[i, 0] = np.cos(angle[i] / 2)
        q[i, 1:] = np.sin(angle[i] / 2) * axis[i] / np.linalg.norm(axis[i])
    return q
