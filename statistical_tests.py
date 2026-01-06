import numpy as np
from scipy.stats import ttest_ind, bootstrap

def significance_test(results_a, results_b, alpha=0.05):
    t_stat, p_value = ttest_ind(results_a, results_b, equal_var=False)
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha
    }

def confidence_interval(data, confidence=0.95):
    res = bootstrap((np.array(data),), np.mean, confidence_level=confidence)
    return res.confidence_interval.low, res.confidence_interval.high
