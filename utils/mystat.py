import numpy as np
from scipy import stats

def cohen_d(x, y):
    """Compute Cohen's d for paired samples"""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def interpret_cohen_d(d):
    """Interpret Cohen's d effect size"""
    if abs(d) < 0.2:
        return "Small"
    elif 0.2 <= abs(d) < 0.8:
        return "Medium"
    else:
        return "Large"

def interpret_rank_correlation(r):
    """Interpret rank-biserial correlation effect size"""
    if abs(r) < 0.1:
        return "Small"
    elif 0.1 <= abs(r) < 0.3:
        return "Medium"
    else:
        return "Large"

def perform_stat_tests(msa1, msa2, label):
    """Performs normality check and paired test, returns results."""
    if len(msa1) == 0 or len(msa2) == 0:
        return {
            'Comparison': label,
            'Num Cases': len(msa1),
            'Shapiro-Wilk test p-value': None,
            'Test used': None,
            'Comparison test p-value': None,
            "Effect Size": None,
            "Effect Size Interpretation": None
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
        effect_size = cohen_d(msa2, msa1)  # Cohen's d for parametric test
        effect_size_interpretation = interpret_cohen_d(effect_size)
    else:
        test_stat, p_value = stats.wilcoxon(msa2, msa1) if len(data_diff) >= 2 else (None, np.nan)
        test_name = "Wilcoxon signed-rank test"
        # Rank correlation (r) for non-parametric test
        if p_value > 0:
            z_value = stats.norm.ppf(p_value / 2)  # Convert p-value to Z-score
            effect_size = abs(z_value) / np.sqrt(len(data_diff))
            effect_size_interpretation = interpret_rank_correlation(effect_size)
        else:
            effect_size = np.nan
            effect_size_interpretation = None


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
        'Effect Size': effect_size,
        'Effect Size Interpretation': effect_size_interpretation,
        'Result of comparison': test_result
    }
