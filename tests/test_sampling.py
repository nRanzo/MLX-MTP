import numpy as np
from mlx_mtp.sampling import SamplerConfig, acceptance_probability, filtered_probabilities, residual_distribution

def test_greedy_distribution_is_point_mass():
    p = filtered_probabilities(np.array([0.0, 2.0, 1.0]), SamplerConfig(temperature=0))
    np.testing.assert_array_equal(p, [0.0, 1.0, 0.0])

def test_top_k_and_top_p_remain_normalized():
    p = filtered_probabilities(np.array([4.0, 3.0, 2.0, 1.0]), SamplerConfig(top_k=2, top_p=.95))
    assert np.isclose(p.sum(), 1.0)
    assert p[2] == p[3] == 0.0

def test_acceptance_is_probability_ratio_in_fp32_range():
    assert acceptance_probability(.25, .5) == .5
    assert acceptance_probability(.9, .1) == 1.0

def test_residual_is_positive_normalized_difference():
    p = np.array([.6, .3, .1]); q = np.array([.2, .2, .3])
    np.testing.assert_allclose(residual_distribution(p, q), [.8, .2, 0.0])
