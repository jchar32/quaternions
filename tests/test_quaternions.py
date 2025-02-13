import numpy as np
import pytest

from . import quaternions as quat


@pytest.fixture
def q():
    return np.array([1, 2, 3, 4])


@pytest.fixture
def q_normalized():
    return np.array([1, 2, 3, 4]) / np.linalg.norm([1, 2, 3, 4])


def test_incorrect_type():
    q = "not a quaternion"
    with pytest.raises(TypeError):
        quat.normalize(q)
    with pytest.raises(TypeError):
        quat.product(q, q)


def test_validate_input_nomatch():
    with pytest.raises(ValueError):
        quat._validate_input(np.array([0]), "notrecognized")


def test_validate_input_quaternion():
    q = np.array([1, 0, 0, 0])
    quat._validate_input(q, "quaternion")
    q = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    quat._validate_input(q, "quaternion")


def test_validate_input_rpy():
    rpy = np.array([0, 0, 0])
    quat._validate_input(rpy, "euler")
    rpy = np.array([[0, 0, 0], [1, 1, 1]])
    quat._validate_input(rpy, "euler")


def test_validate_input_rotmat():
    rotmat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    quat._validate_input(rotmat, "rotmat")
    rotmat = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    quat._validate_input(rotmat, "rotmat")
    with pytest.raises(ValueError):
        rotmat = np.array([[1, 2, 3], [3, 2, 1]])
        quat._validate_input(rotmat, "rotmat")
    rotmat = rotmat = np.random.rand(2, 3, 3)
    quat._validate_input(rotmat, "rotmat")
    rotmat = np.random.rand(2, 3, 5)
    with pytest.raises(ValueError):
        quat._validate_input(rotmat, "rotmat")


# TODO: Add functionality to accept list and convert to ndarray
def test_validate_input_invalid_type():
    invalid = [1, 2, 3, 4]
    with pytest.raises(TypeError):
        quat._validate_input(invalid, "quaternion")
    invalid = 1.0
    with pytest.raises(TypeError):
        quat._validate_input(invalid, "rpy")
    invalid = int(1)
    with pytest.raises(TypeError):
        quat._validate_input(invalid, "rotmat")


def test_validate_input_invalid_shape():
    invalid = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        quat._validate_input(invalid, "quaternion")
    invalid = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        quat._validate_input(invalid, "quaternion")
    invalid = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        quat._validate_input(invalid, "rotmat")
    with pytest.raises(ValueError):
        invalid = np.random.rand(2, 3, 4, 5, 5)
        quat._validate_input(invalid, "quaternion")


def test_product(q, q_normalized):
    q_inv = np.array([q_normalized[0], -q_normalized[1], -q_normalized[2], -q_normalized[3]])
    result = quat.product(q_normalized, q_inv)
    identity = np.array([1, 0, 0, 0])
    assert np.allclose(result, identity), "Product of quaternion and its inverse is not identity"

    q_5 = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        quat.product(q_5, q_5)
    q_nd = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    assert q_nd.shape == quat.product(q_nd, q_nd).shape

    zero = np.array([0, 0, 0, 0])
    result = quat.product(q, zero)
    assert np.allclose(result, zero), "Product with zero quaternion is not zero"

    q1 = np.array([1, 2, 3, 4])
    q2 = np.array([5, 6, 7, 8])
    assert not np.allclose(quat.product(q1, q2), quat.product(q2, q1)), "Quaternion multiplication is commutative"

    q_scalar_last = np.array([[0, 0, 0, 1]])
    p_scalar_last = np.array([[0, 0, 0, 2]])
    assert np.allclose(quat.product(q_scalar_last, p_scalar_last, scalarLast=True), np.array([2, 0, 0, 0]))


def test_normalize(q, q_normalized):
    q_norm_test = quat.normalize(q)
    assert np.array_equal(q_norm_test, q_normalized), "Length of normalized quaternion is not 1"

    q_5 = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        quat.normalize(q_5)
    q_nd = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    assert q_nd.shape == quat.normalize(q_nd).shape

    q = np.array([0, 0, 0, 0])
    with pytest.raises(ArithmeticError):
        quat.normalize(q)
    q = np.zeros((10, 4))
    with pytest.raises(ArithmeticError):
        quat.normalize(q)


def test_quat_rotate():
    q_identity = np.array([1, 0, 0, 0])
    p = np.array([1, 0, 0])
    rotated_p = quat.quat_rotate(q_identity, p)
    assert np.allclose(rotated_p, p), "Identity rotation should leave the vector unchanged"

    q_identity = np.array([1, np.nan, 0, 0])
    p = np.array([1, 0, 0])
    with pytest.raises(ValueError):
        quat.quat_rotate(q_identity, p)

    q = np.array([[0, 0, 0, 1], [0, 0, 0, 1]])
    v = np.array([1, 0, 0])
    with pytest.raises(ValueError):
        quat.quat_rotate(q, v)

    # 90-degree rotation around the z-axis
    q_90_z = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    p = np.array([1, 0, 0])
    rotated_p = quat.quat_rotate(q_90_z, p)
    assert np.allclose(rotated_p, [0, 1, 0]), "90-degree rotation around z-axis is incorrect"

    # 180-degree rotation around the y-axis
    q_180_y = np.array([np.cos(np.pi / 2), 0, np.sin(np.pi / 2), 0])
    p = np.array([1, 0, 0])
    rotated_p = quat.quat_rotate(q_180_y, p)
    assert np.allclose(rotated_p, [-1, 0, 0]), "180-degree rotation around y-axis is incorrect"

    # 180-degree rotation around the x-axis
    q_180_x = np.array([np.cos(np.pi / 2), np.sin(np.pi / 2), 0, 0])
    p = np.array([0, 1, 0])
    rotated_p = quat.quat_rotate(q_180_x, p)
    assert np.allclose(rotated_p, [0, -1, 0]), "180-degree rotation around x-axis is incorrect"

    q_batch = np.array([[1, 0, 0, 0.1], [1, 0, 0, 0.1], [1, 0, 0, 0.1]])
    v_batch = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotated_batch = quat.quat_rotate(q_batch, v_batch)
    assert rotated_batch.shape == (3, 3), "Batch rotation should have correct shape"


def test_conjugate(q, q_normalized):
    q_conjugate = quat.conjugate(q)
    expected_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])
    assert np.allclose(q_conjugate, expected_conjugate), "Conjugate failed for q"

    # conjugate of product of two quaternions is the product of conjugates in reverse order
    p_normalized = np.array([4, 3, 6, 2]) / np.linalg.norm([4, 3, 6, 2])
    q_conjugate_communative = quat.conjugate(quat.product(q_normalized, p_normalized))
    q_communiative = quat.product(quat.conjugate(p_normalized), quat.conjugate(q_normalized))
    assert np.allclose(q_conjugate_communative, q_communiative), "Conjugate failed for normalized q"


def test_inverse(q, q_normalized):
    # Test inverse for a non-normalized quaternion
    q_inv = quat.inverse(q)
    q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])  # Conjugate of q
    q_norm_sq = np.linalg.norm(q) ** 2
    expected_inverse = q_conjugate / q_norm_sq

    # Check if the inverse is correct by asserting equality
    assert np.allclose(q_inv, expected_inverse), f"Inverse failed for q: {q}"

    # Test inverse for a normalized quaternion
    q_inv_normalized = quat.inverse(q_normalized)
    q_conjugate_normalized = np.array([q_normalized[0], -q_normalized[1], -q_normalized[2], -q_normalized[3]])
    expected_inverse_normalized = q_conjugate_normalized  # Norm is 1 for normalized quaternion

    assert np.allclose(q_inv_normalized, expected_inverse_normalized), "Inverse failed for normalized q"


def test_exponential(q, q_normalized):
    # Test exponential for a non-normalized quaternion
    q_exp = quat.exponential(q)
    q0 = q[0]
    qv = q[1:4]
    qv_norm = np.linalg.norm(qv)
    eq0 = (np.exp(q0) * np.cos(qv_norm),)
    eqv = np.exp(q0) * qv * np.sin(qv_norm) / qv_norm
    expected_exp = np.concatenate([eq0, eqv])

    # Check if the exponential is correct
    assert np.allclose(q_exp, expected_exp), f"Exponential failed for q: {q}"

    # Test exponential for a normalized quaternion
    q_exp_normalized = quat.exponential(q_normalized)
    q0_norm = q_normalized[0]
    qv_normalized = q_normalized[1:4]
    qv_norm_val = np.linalg.norm(qv_normalized)
    eq0_norm = (np.exp(q0_norm) * np.cos(qv_norm_val),)
    eqv_norm = np.exp(q0_norm) * qv_normalized * np.sin(qv_norm_val) / qv_norm_val
    expected_exp_normalized = np.concatenate([eq0_norm, eqv_norm])

    assert np.allclose(q_exp_normalized, expected_exp_normalized), "Exponential failed for normalized q"


# Test for logarithm function
def test_logarithm(q, q_normalized):
    # Test logarithm for a non-normalized quaternion
    q_log = quat.logarithm(q)
    q0 = q[0]
    qv = q[1:4]
    q_norm = np.linalg.norm(q)
    qv_norm = np.linalg.norm(qv)
    logq0 = (np.log(q_norm),)
    logqv = qv * np.arccos(q0 / q_norm) / qv_norm
    expected_log = np.concatenate([logq0, logqv])

    # Check if the logarithm is correct
    assert np.allclose(q_log, expected_log), f"Logarithm failed for q: {q}"

    # Test logarithm for a normalized quaternion
    q_log_normalized = quat.logarithm(q_normalized)
    q0_norm = q_normalized[0]
    qv_norm_normalized = q_normalized[1:4]
    q_norm_normalized = np.linalg.norm(q_normalized)

    expected_log_normalized_q0 = (np.log(q_norm_normalized),)
    expected_log_normalized_qv = qv_norm_normalized * np.arccos(q0_norm / q_norm_normalized) / np.linalg.norm(qv_norm_normalized)

    expected_log_normalized = np.concatenate([expected_log_normalized_q0, expected_log_normalized_qv])
    assert np.allclose(q_log_normalized, expected_log_normalized), "Logarithm failed for normalized q"


# Edge case test: identity quaternion
def test_identity_quaternion_computations():
    identity_q = np.array([1, 0, 0, 0])

    assert quat.product(identity_q, identity_q).all() == identity_q.all(), "Product of identity quaternion failed"

    assert np.allclose(quat.normalize(identity_q), identity_q), "Normalize of identity quaternion failed"

    # Test inverse of identity quaternion should be identity quaternion
    assert np.allclose(quat.inverse(identity_q), identity_q), "Inverse of identity quaternion failed"

    # Test exponential of identity quaternion should be [1, 0, 0, 0]
    assert np.allclose(quat.exponential(identity_q), identity_q), "Exponential of identity quaternion failed"

    # Test logarithm of identity quaternion should be [0, 0, 0, 0]
    assert np.allclose(quat.logarithm(identity_q), np.array([0, 0, 0, 0])), "Logarithm of identity quaternion failed"


# Edge case test: pure quaternion (no scalar part)
def test_pure_quaternion():
    pure_q = np.array([0, 1, 0, 0])
    # Test inverse of pure quaternion
    q_inv = quat.inverse(pure_q)
    q_conjugate = np.array([0, -1, -0, -0])
    q_norm_sq = np.linalg.norm(pure_q) ** 2
    expected_inv_pure = q_conjugate / q_norm_sq
    assert np.allclose(q_inv, expected_inv_pure), "Inverse of pure quaternion failed"

    # Test exponential of pure quaternion
    q_exp = quat.exponential(np.array([0, 1, 0, 0]))
    assert np.allclose(q_exp, np.array([np.cos(1), np.sin(1), 0, 0])), "Exponential of pure quaternion failed"

    # Test logarithm of pure quaternion
    q_log = quat.logarithm(np.array([0, 1, 0, 0]))
    assert np.allclose(q_log, np.array([0, np.pi / 2, 0, 0])), "Logarithm of pure quaternion failed"


def test_to_scalar_first():
    scalar_is_last = np.array([1, 2, 3, 4])
    scalar_is_first = np.array([4, 1, 2, 3])
    assert np.allclose(quat.to_scalar_first(scalar_is_last), scalar_is_first), "Scalar-last to scalar-first conversion failed"

    scalar_is_last = np.ones((10, 4)) * np.array([1, 2, 3, 4])
    scalar_is_first = np.ones((10, 4)) * np.array([4, 1, 2, 3])
    assert np.allclose(quat.to_scalar_first(scalar_is_last), scalar_is_first), "Scalar-last to scalar-first conversion failed"


def test_is_unit(q):
    assert not quat.is_unit(q), "is_unit failed for non-unit quaternion"
    assert quat.is_unit(q / np.linalg.norm(q)), "is_unit failed for unit quaternion"


def test_is_pure(q):
    assert not quat.is_pure(q), "is_pure failed for non-pure quaternion"
    assert quat.is_pure(np.array([0, 1, 2, 3])), "is_pure failed for pure quaternion"


def test_is_real(q):
    assert not quat.is_real(q), "is_real failed for non-real quaternion"
    assert quat.is_real(np.array([1, 0, 0, 0])), "is_real failed for real quaternion"


def test_identity():
    assert np.allclose(quat.identity(), np.array([1, 0, 0, 0])), "identity not producing identity quaternion"


def test_from_angles():
    # Test for invalid input
    with pytest.raises(ValueError):
        quat.from_angles(np.array([np.nan, 0, 0, 0]))

    # Test for Euler angles
    angles = np.array([0, 0, 0])
    q = quat.from_angles(angles, "xyz")
    assert np.allclose(q, np.array([1, 0, 0, 0])), "from_angles failed for Euler angles"

    # Rotation around x
    angle = np.array([np.pi / 2, 0, 0])
    q = quat.from_angles(angle, "xyz")
    assert np.allclose(q, np.array([np.cos(angle[0] / 2), np.cos(angle[0] / 2), 0, 0])), "from_angles failed for single angle of 90deg which should produce q=(cos(90/2), cos(90/2), 0, 0)"

    # Rotation Around y
    angle = np.array([0, np.pi / 2, 0])
    q = quat.from_angles(angle, "xyz")
    assert np.allclose(q, np.array([np.cos(angle[1] / 2), 0, np.cos(angle[1] / 2), 0])), "from_angles failed for single angle of 90deg which should produce q=(cos(90/2),0, cos(90/2), 0)"

    # Rotation Around z
    angle = np.array([0, 0, np.pi / 2])
    q = quat.from_angles(angle, "xyz")
    assert np.allclose(q, np.array([np.cos(angle[-1] / 2), 0, 0, np.cos(angle[-1] / 2)])), "from_angles failed for single angle of 90deg which should produce q=(cos(90/2),0,0 cos(90/2))"

    # Rotation around y and z
    angle = np.array([0, np.pi / 4, np.pi / 2])
    q = quat.from_angles(angle, "xyz")
    assert np.allclose(q, np.array([0.65328148, 0.27059805, 0.27059805, 0.65328148])), "from_angles failed for single angle of 90deg which should produce q=(cos(45/2),0,0 cos(90/2))"

    angles = np.zeros(3)
    for order in ["xyz", "zyx", "yxz", "zxy", "yzx", "xzy"]:
        result = quat.from_angles(angles, order=order)
        assert np.allclose(result, np.array([1, 0, 0, 0])), f"from_angles {order} failed for identity"


def test_from_angles_nonfinite():
    # Test that non-finite inputs raise an error.
    with pytest.raises(ValueError):
        quat.from_angles(np.array([np.nan, 0, 0]))
    with pytest.raises(ValueError):
        quat.from_angles(np.array([np.inf, 0, 0]))


def test_to_angles():
    # Test for Euler angles
    q = np.array([1, 0, 0, 0])
    angles = quat.to_angles(q)
    assert np.allclose(angles, np.array([0, 0, 0])), "to_angles failed for Euler angles"

    q = np.array([0, 0, 0, 1])
    angles = quat.to_angles(q, scalarLast=True)
    assert np.allclose(angles, np.array([0, 0, 0])), "to_angles failed for Euler angles"

    q = np.array([np.pi / 2, np.pi / 2, 0, 0]) / np.linalg.norm([np.pi / 2, np.pi / 2, 0, 0])
    angles = quat.to_angles(q)
    assert np.allclose(angles, np.array([np.pi / 2, 0, 0])), "to_angles failed for single angle of 90deg which should produce q=(cos(45), cos(45), 0, 0)"

    qnan = np.array([np.nan, 0, 0, 0])
    with pytest.raises(ValueError):
        quat.to_angles(qnan)

    with pytest.raises(ValueError):
        quat.to_angles(np.array([1, 0, 0, 0]), order="ZYZ")

    # test gimbal lock
    q = np.array([0, 0.7071068, 0, 0.7071068])
    angles = quat.to_angles(q, order="xyz")
    assert np.allclose(angles, np.array([np.pi, -np.pi / 2, 0])), "to_angles failed for gimbal lock"

    # test multiple angle sets
    q = np.array([[1, 0, 0, 0], [0, 0.7071068, 0, 0.7071068], [0.5963678, 0.6903455, -0.1530459, 0.3799282]])
    expected_angles = np.array([[0, 0, 0], [np.pi, -np.pi / 2, 0], [np.pi / 2, -np.pi / 4, np.pi / 9]])
    result = quat.to_angles(q, order="xyz")
    assert np.allclose(result, expected_angles, atol=1e-6), "to_angles failed for multiple quaternions"


def test_to_rotmat():
    # Identity quaternion
    q_eye = np.array([1, 0, 0, 0])
    R = quat.to_rotmat(q_eye, scalarLast=False, homogenous=True)
    expected_R = np.eye(3)
    assert np.allclose(R, expected_R), "Identity quaternion should produce identity rotation matrix"

    q = np.array([0, 0, 0, 1])  # Identity rotation in scalar-last format
    R = quat.to_rotmat(q, scalarLast=True, homogenous=True)
    expected_R = np.eye(3)
    assert np.allclose(R, expected_R), "Scalar-last identity quaternion should produce identity rotation matrix"

    # Identity rotation
    R = quat.to_rotmat(q_eye, scalarLast=False, homogenous=False)
    expected_R = np.eye(3)
    assert np.allclose(R, expected_R), "Non-homogeneous identity quaternion should produce identity rotation matrix"

    q90xy = np.array([0.5, 0.5, 0.5, 0.5])  # 90-degree rotation about x-axis then y axis
    R_expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # Expected rotation matrix
    R = quat.to_rotmat(q90xy, scalarLast=False, homogenous=True)
    assert np.allclose(R, R_expected), "Quaternion -> Matrix -> Quaternion should be consistent"

    q = np.array([2, 0, 0, 0])  # Non-unit quaternion
    R = quat.to_rotmat(q, scalarLast=False, homogenous=True)
    expected_R = np.eye(3)
    assert np.allclose(R, expected_R), "Non-unit quaternion should normalize internally and produce identity matrix"

    q_array = np.array([[1, 0, 0, 0], [0.7071, 0.7071, 0, 0]])  # Identity and 90-degree x-rotation
    R = np.array(quat.to_rotmat(q_array, scalarLast=False, homogenous=True))
    assert R.shape == (2, 3, 3), "Rotation matrices should have correct shape for multiple quaternions"
    assert np.allclose(R[0], np.eye(3)), "First quaternion should produce identity matrix"


def test_from_rotmat():
    R_identity = np.eye(3)
    expected_quat = np.array([1, 0, 0, 0])
    assert np.allclose(quat.from_rotmat(R_identity), expected_quat, atol=1e-6)

    R_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # 90-degree rotation about x-axis
    expected_quat = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
    assert np.allclose(quat.from_rotmat(R_x), expected_quat, atol=1e-6)

    R_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # 90-degree rotation about y-axis
    expected_quat = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
    assert np.allclose(quat.from_rotmat(R_y), expected_quat, atol=1e-6) or np.allclose(quat.from_rotmat(R_y), -expected_quat, atol=1e-6)

    R_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90-degree rotation about z-axis
    expected_quat = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    assert np.allclose(quat.from_rotmat(R_z), expected_quat, atol=1e-6)

    R_arbitrary = np.array([[0.0000000, -0.7071068, 0.7071068], [0.9396926, -0.2418448, -0.2418448], [0.3420202, 0.6644630, 0.6644630]])  # rotoation xyz = X:20, Y45, Z90

    expected_quat = np.array([0.5963678, 0.3799282, 0.1530459, 0.6903455])
    assert np.allclose(quat.from_rotmat(R_arbitrary), expected_quat, atol=1e-6)

    R_x = np.array([[1, 0, 0], [0, 0.7071068, -0.7071068], [0, 0.7071068, 0.7071068]])  # 90-degree rotation about x-axis
    R_y = np.array([[0.7071068, 0, 0.7071068], [0, 1, 0], [-0.7071068, 0, 0.7071068]])
    R_z = np.array([[0.7071068, -0.7071068, 0], [0.7071068, 0.7071068, 0], [0, 0, 1]])  # 90-degree rotation about z-axis
    rotations = np.concat([R_x[np.newaxis, :], R_y[np.newaxis, :], R_z[np.newaxis, :]])

    expected_quats = np.array([[0.9238795, 0.3826834, 0, 0], [0.9238795, 0, 0.3826834, 0], [0.9238795, 0, 0, 0.3826834]])
    computed_quats = quat.from_rotmat(rotations)

    for cq, eq in zip(computed_quats, expected_quats):
        assert np.allclose(cq, eq, atol=1e-6)

    R_neg_trace = np.array([[-0.5, -0.5, 0.7071], [-0.5, -0.5, -0.7071], [0.7071, -0.7071, 0]])
    rotations = np.concat([R_x[np.newaxis, :], R_neg_trace[np.newaxis, :], R_z[np.newaxis, :]])
    expected_quats = np.array([[0.9238795, 0.3826834, 0, 0], [0.0, 0.63245432, -0.63245432, 0.44721703], [0.9238795, 0, 0, 0.3826834]])
    assert np.allclose(quat.from_rotmat(rotations), expected_quats, atol=1e-6)

    with pytest.raises(ValueError):
        quat.from_rotmat(np.eye(4))  # Invalid shape

    with pytest.raises(ValueError):
        quat.from_rotmat(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]]))  # Not a valid rotation matrix


def test_from_axis_angle(q, q_normalized):
    ax = np.array([[0, 0, 1, 0]])  # Zero angle about z-axis
    q = quat.from_axis_angle(ax, angleFirst=False)
    expected_q = np.array([[1, 0, 0, 0]])  # Identity quaternion
    assert np.allclose(q, expected_q), "Zero angle should result in identity quaternion"

    ax = np.array([[0, 0, 1, np.pi]])  # 180-degree rotation about z-axis
    q = quat.from_axis_angle(ax, angleFirst=False)
    expected_q = np.array([[0, 0, 0, 1]])
    assert np.allclose(q, expected_q), "180-degree z-axis rotation should produce correct quaternion"

    ax = np.array([[np.pi, 0, 0, 1]])  # 180-degree rotation about z-axis, angle first
    q = quat.from_axis_angle(ax, angleFirst=True)
    expected_q = np.array([[0, 0, 0, 1]])
    assert np.allclose(q, expected_q), "Angle-first input should produce correct quaternion"

    ax = np.array([[0, 0, 2, np.pi]])  # 180-degree rotation about a non-unit z-axis
    q = quat.from_axis_angle(ax, angleFirst=False)

    expected_q = np.array([[0, 0, 0, 1]])  # Should normalize the axis
    assert np.allclose(q, expected_q), "Non-unit axis should be normalized"

    ax = np.array([[0, 0, 0, 0]])  # all zeros
    q = quat.from_axis_angle(ax, angleFirst=True)
    expected_q = np.array([[1, 0, 0, 0]])  # Should normalize the axis
    assert np.allclose(q, expected_q), "All zero resturns identity quaternion"

    ax = np.array(
        [
            [0, 0, 0, 1],  # Identity rotation
            [np.pi / 2, 1, 0, 0],  # 90-degree x-axis rotation
            [np.pi, 0, 1, 0],  # 180-degree y-axis rotation
        ]
    )
    q = quat.from_axis_angle(ax, angleFirst=True)
    expected_q = np.array(
        [
            [1, 0, 0, 0],  # Identity quaternion
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0],  # 90-degree x-axis rotation
            [0, 0, 1, 0],  # 180-degree y-axis rotation
        ]
    )
    assert np.allclose(q, expected_q), "Batch processing should produce correct quaternions"

    num_samples = 1000
    ax = np.hstack(
        (
            np.random.rand(num_samples, 3) - 0.5,  # Random axes
            np.random.rand(num_samples, 1) * 2 * np.pi,  # Random angles
        )
    )
    q = quat.from_axis_angle(ax, angleFirst=False)
    assert q.shape == (num_samples, 4), "Output should have correct shape for large batch"

    ax = np.array([[0, 0, 1, np.pi / 2]])  # 90-degree rotation about z-axis
    q = quat.from_axis_angle(ax, angleFirst=False)
    expected_q = np.array([[np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]])
    assert np.allclose(q, expected_q), "Single input should produce correct quaternion"


def test_yaw_quat():
    q = np.array([0.5, 0.5, 0, 0.7071068])
    expected_yaw = 90
    assert np.allclose(quat.yaw(q), expected_yaw), "Yaw calculation failed"

    q = np.array([[0.5, 0.5, 0, 0.7071068], [0.5, 0.5, 0, 0.7071068]])
    expected_yaw = np.array([90, 90])
    assert np.allclose(quat.yaw(q), expected_yaw), "Yaw calculation failed"
