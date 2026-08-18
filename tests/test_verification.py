import numpy as np
import pytest
from mlx_mtp.verification import assert_close, max_abs_diff

def test_max_abs_diff_and_tolerance():
    assert max_abs_diff([1, 2], [1, 2.00001]) < 1e-4
    assert assert_close([1, 2], [1, 2.00001]) < 1e-4
    with pytest.raises(AssertionError, match="logits"):
        assert_close([1], [1.1], name="logits")
