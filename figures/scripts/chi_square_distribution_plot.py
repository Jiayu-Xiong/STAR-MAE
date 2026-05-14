# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from scipy.stats import chi2, chisquare

def chi_square_test(prefix, t, fit_count=None, fit_start=0, alpha=0.05):
    """
    Perform a Chi-Square goodness-of-fit test on the input data `all_concat`,
    only considering data within the [1e-2, 2] interval,
    and apply corresponding truncation to the Chi-Square distribution within [1e-2, 2]
    to determine if it follows the (truncated) Chi-Square distribution.
    
    Args:
        prefix (str): Directory prefix where the .npy files are located.
        t (int): Number of files to process.
        alpha (float, optional): Significance level for the test. Defaults to 0.05.
    """

    all_mse_values_list = []
    for i in range(t):
        file_path = prefix + f"mse_distribution_{i}.npy"
        mse_array = np.load(file_path)
        all_mse_values_list.append(mse_array)
    all_concat = np.concatenate(all_mse_values_list)

    lower_bound = 1e-2
    upper_bound = 40.0
    filter = all_concat[(all_concat >= lower_bound) & (all_concat <= upper_bound)]
    upper_bound = 2.0
    filter = all_concat[(all_concat >= lower_bound) & (all_concat <= upper_bound)]
    if fit_count is None:
        fit_count = len(filter)
    fit_end = fit_start + fit_count
    if fit_start < 0 or fit_count <= 0 or fit_end > len(filter):
        raise ValueError(
            f'Invalid deterministic fit window: start={fit_start}, count={fit_count}, '
            f'available={len(filter)}')
    filter_low = filter[fit_start:fit_end]
    print(f'fit_count={len(filter_low)}')
    result = analyze_distribution(filter_low)
    print(result)
    fig, ax = plot_distribution_fits(filter, result)

    # assert False

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def plot_distribution_fits(orig_data, fit_results, figsize=(8, 2.5), bins=300, xlim=None, density=True):
    """
    Plot a histogram of the original data with fitted distribution curves.
    
    Parameters:
    -----------
    orig_data : numpy.ndarray
        Original data array to be plotted as a histogram.
    fit_results : dict
        Dictionary containing distribution test results from the analyze_distribution function.
    figsize : tuple, optional
        Figure size in inches (width, height). Defaults to (8, 2.5).
    bins : int, optional
        Number of bins for the histogram. Defaults to 300.
    xlim : tuple, optional
        Limits for the x-axis (min, max). If None, will be determined from the data.
    density : bool, optional
        If True, plot density instead of counts. Defaults to True.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The created axes object.
    """
    plt.style.use('seaborn-v0_8-paper')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot histogram
    hist_color = '#E6E6E6'
    sns.histplot(data=orig_data, bins=bins, color=hist_color, alpha=0.6, 
                stat='density' if density else 'count', label='Original Data')
    
    # Define color palette for distribution curves
    colors = ['#1ba784', '#f28e16', '#ed556a', '#2f90b9', '#b0a4e3', '#000000']
    
    # Set x range for distribution curves
    if xlim is None:
        xlim = (max(0, orig_data.min()), min(orig_data.max(), np.percentile(orig_data, 99.5)))
    x = np.linspace(xlim[0], xlim[1], 200)
    
    # Sort distributions by AIC
    sorted_results = sorted(fit_results['distribution_tests'], 
                            key=lambda x: x['aic'])
    
    # Plot each distribution
    for idx, result in enumerate(sorted_results):
        dist_name = result['distribution']
        params = result['params']
        p_value = result['p_value']
        aic = result['aic']
        
        # Get distribution object
        dist = getattr(stats, dist_name)
        
        # Calculate PDF
        pdf = dist.pdf(x, *params)
        
        # Plot PDF
        label = f"{dist_name.capitalize()}\n(p={p_value:.3f}, AIC={aic:.1f})"
        ax.plot(x, pdf, color=colors[idx], linewidth=2, label=label, alpha=0.8)
    
    # Remove labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Move legend to upper right inside the plot
    ax.legend(loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.ticklabel_format(axis='y', style='sci', scilimits=(1, 1))
    # Minimize white space
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save as PDF with minimal margins
    plt.savefig('distribution_fits.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    
    return fig, ax

# Example usage:
"""
# Assuming you have your original data in 'data' and fit results in 'results'
fig, ax = plot_distribution_fits(data, results)
plt.show()

# To save the plot:
# fig.savefig('distribution_fits.png', dpi=300, bbox_inches='tight')
"""

def analyze_distribution(data):
    """
    Analyze the distribution of the data through multiple statistical tests
    and distribution fittings.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data array to analyze.
    
    Returns:
    --------
    dict
        Dictionary containing basic statistics and distribution test results.
    """
    # Basic statistics
    mean = np.mean(data)
    std = np.std(data)
    skew = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    # Test multiple distributions
    distributions = [
        ('chi2', stats.chi2),
        ('gamma', stats.gamma),
        ('lognorm', stats.lognorm),
        ('weibull_min', stats.weibull_min),
        ('invgauss', stats.invgauss)
        # ('norm', stats.norm)
    ]
    
    results = []
    for name, distribution in distributions:
        # Fit distribution parameters
        params = distribution.fit(data)
        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.kstest(data, distribution.cdf, params)
        # Calculate AIC
        log_likelihood = np.sum(distribution.logpdf(data, *params))
        k = len(params)
        n = len(data)
        aic = 2 * k - 2 * log_likelihood
        
        results.append({
            'distribution': name,
            'params': params,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'aic': aic
        })
        
    # Sort results by AIC
    results.sort(key=lambda x: x['aic'])
    
    # Plot histogram and fitted distributions
    # plt.figure(figsize=(12, 6))
    # plt.hist(data, bins=50, density=True, alpha=0.6, label='Data')
    
    # x = np.linspace(min(data), max(data), 100)
    # for result in results[:3]:  # Plot top 3 distributions
    #     dist = getattr(stats, result['distribution'])
    #     plt.plot(x, dist.pdf(x, *result['params']), 
    #             label=f"{result['distribution']}")
    # 
    # plt.legend()
    # plt.title('Data Distribution with Fitted Curves')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    
    return {
        'basic_stats': {
            'mean': mean,
            'std': std,
            'skewness': skew,
            'kurtosis': kurtosis
        },
        'distribution_tests': results
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Fit distributions for user-provided MSE .npy results.')
    parser.add_argument('--prefix', required=True, help='Directory containing mse_distribution_{i}.npy files.')
    parser.add_argument('--runs', type=int, required=True, help='Number of files to process.')
    parser.add_argument('--fit-count', type=int, default=None, help='Deterministic number of filtered samples used for fitting.')
    parser.add_argument('--fit-start', type=int, default=0, help='Deterministic start offset in the filtered sample array.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chi_square_test(args.prefix, args.runs, fit_count=args.fit_count, fit_start=args.fit_start, alpha=args.alpha)
