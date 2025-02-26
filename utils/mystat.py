import numpy as np
from scipy import stats

def perform_stat_tests(msa1, msa2, label):
    """Performs normality check and paired test, returns results."""
    if len(msa1) == 0 or len(msa2) == 0:
        return {
            'Comparison': label,
            'Num Cases': len(msa1),
            'Shapiro-Wilk test p-value': None,
            'Test used': None,
            'Comparison test p-value': None
        }

    # Compute differences
    data_diff = msa2 - msa1

    # Interpret significance
    alpha = 0.05  # Common significance level

    # Check normality (only if at least 3 samples)
    shapiro_p = stats.shapiro(data_diff).pvalue if len(data_diff) >= 3 else np.nan

    # Choose appropriate test
    if shapiro_p > 0.05:
        test_stat, p_value = stats.ttest_rel(msa2, msa1)  # Paired t-test
        test_name = "Paired t-test"
    else:
        test_stat, p_value = stats.wilcoxon(msa2, msa1) if len(data_diff) >= 2 else (None, np.nan)
        test_name = "Wilcoxon signed-rank test"

    if p_value < alpha:
        test_result = "Significant difference"    
    else:
        test_result = "No significant difference"

    return {
        'Comparison': label,
        'Num Cases': len(msa1),
        'Shapiro-Wilk test p-value': shapiro_p,
        'Test used': test_name,
        'Comparison test p-value': p_value,
        'Result of comparison': test_result
    }
