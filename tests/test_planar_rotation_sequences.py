import numpy as np
import pytest
from spatialmath import SO2, SE2


@pytest.mark.parametrize("cls", [SO2, SE2])
@pytest.mark.parametrize("count", [1, 2, 5])
@pytest.mark.parametrize("stacked", [False, True])
def test_rotation_components(cls, count, stacked):
    angles = np.linspace(-1.2, 0.7, count)
    values = [
        cls(a) if cls is SO2 else cls(i + 1, -i - 2, a) for i, a in enumerate(angles)
    ]
    matrices = [x.A.copy() for x in values]
    poses = cls(np.stack(matrices) if stacked else matrices)
    before = [x.copy() for x in poses.data]
    result = poses.R
    expected = np.array(
        [[[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]] for a in angles]
    )
    if count == 1:
        expected = expected[0]
    np.testing.assert_allclose(result, expected, atol=1e-15)
    assert result.shape == ((2, 2) if count == 1 else (count, 2, 2))
    for actual, original in zip(poses.data, before):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("cls", [SO2, SE2])
def test_sequence_rotation_matches_singletons(cls):
    values = [cls(a) if cls is SO2 else cls(2, -3, a) for a in [-0.7, 0, 1.1]]
    poses = cls(values)
    np.testing.assert_array_equal(poses.R, np.stack([x.R for x in values]))
