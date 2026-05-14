# Author: Jiayu Xiong
# Reference: STAR-MAE / Distribution-aware Loss Reweighting (DLR), PR-D-25-06353_R2.
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # For smoothing curves
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import os
import argparse
from scipy import stats


def plot_average_mse_distribution_all(
    file_prefix_list,  # List of file prefixes, e.g., ['mse_distribution_A', 'mse_distribution_B', ...]
    t,                 # Number of .npy files per prefix
    names,             # Legend names corresponding to file_prefix_list, e.g., ['Group A', 'Group B', ...]
    bins=500,          # Number of histogram bins
    colors=None,       # List of colors for each file_prefix
    save_path='mse_distribution_all.pdf'
):
    """
    Reads evaluation result files under multiple file_prefixes, each containing t .npy files,
    computes the average histogram statistics for each prefix, and plots them on the same graph.
    Each bin's groups are plotted in order of descending count to ensure the highest values are at the bottom.
    Uses the provided colors for each file_prefix and includes a legend with the provided names.

    :param file_prefix_list: list of str, each element is a file prefix (excluding the _i.npy part)
    :param t: int, number of evaluation runs per file prefix
    :param names: list of str, legend names corresponding to file_prefix_list
    :param bins: int, number of histogram bins
    :param colors: list of str, colors for each file_prefix, e.g., ['#b0a4e3', '#FF5733', ...]
    :param save_path: str, path to save the final plot (PDF format)
    """
    if colors is None:
        # Default colors, reused if not enough colors are provided
        colors = ['#b0a4e3', '#FF5733', '#33FF57', '#3357FF', '#FF33A6']

    # Store average histogram counts for each group, maintaining the order of file_prefix_list
    group_avg_counts = []
    # Define a unified range for all histograms, without w=0 approx domain
    min_val, max_val = 5e-3, 1.25
    # Create uniform bin edges
    bins_array = np.linspace(min_val, max_val, bins + 1)
    dx = bins_array[1] - bins_array[0]
    bin_centers = 0.5 * (bins_array[:-1] + bins_array[1:])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, prefix in enumerate(file_prefix_list):
        all_mse_values_list = []
        for i in range(t):
            file_path = os.path.join(prefix, f"mse_distribution_{i}.npy")
            mse_array = np.load(file_path)
            all_mse_values_list.append(mse_array)

        # Concatenate all data for statistical information
        all_concat = np.concatenate(all_mse_values_list)
        print(f"[{prefix}] mean: {all_concat.mean():.4f}  max: {all_concat.max():.4f}")

        # Compute histogram counts for each evaluation run and sum them
        sum_counts = np.zeros(bins)
        for mse_array in all_mse_values_list:
            counts, _ = np.histogram(mse_array, bins=bins_array)
            sum_counts += counts

        # Calculate average counts
        avg_counts = sum_counts / t
        group_avg_counts.append(avg_counts)
        # Apply Gaussian smoothing to the counts
        smooth_counts = gaussian_filter1d(avg_counts + 1e4, sigma=1)

        # Plot the smoothed curve
        ax.plot(bin_centers, smooth_counts, color=colors[idx % len(colors)],
                linewidth=2, zorder=100)

    num_groups = len(group_avg_counts)

    # Draw rectangles for each bin and group
    for j, center in enumerate(bin_centers):
        # Collect counts for each group at this bin
        group_vals = [(idx, group_avg_counts[idx][j]) for idx in range(num_groups)]
        # Sort groups by descending count to plot the largest first
        group_sorted = sorted(group_vals, key=lambda x: x[1], reverse=True)
        for order, (g_idx, val) in enumerate(group_sorted):
            # Draw rectangle: x starts at (center - dx/2), y starts at 0
            rect = patches.Rectangle(
                (center - dx / 2, 0),  # Bottom-left corner
                dx,                    # Width
                val,                   # Height
                facecolor=colors[g_idx % len(colors)],
                alpha=0.9,
                edgecolor='black',
                zorder=order  # Lower zorder is drawn first (at the bottom)
            )
            ax.add_patch(rect)
            # Add a marker at the top of the rectangle
            ax.plot(center, val, marker='o', color=colors[g_idx % len(colors)], markersize=4, zorder=order + 10)

    # Set axis limits
    ax.set_xlim(min_val, max_val)
    max_y = max(np.max(avg) for avg in group_avg_counts)
    ax.set_ylim(0, max_y * 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Use scientific notation for the y-axis
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))

    # Create legend using colored patches
    legend_handles = []
    for idx, name in enumerate(names):
        patch = patches.Patch(color=colors[idx % len(colors)], label=name, alpha=0.7)
        legend_handles.append(patch)
    ax.legend(handles=legend_handles)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()

    print(f"[Info] Average distribution histogram saved to: {save_path}")


def plot_average_mse_distribution_all_norm(
    file_prefix_list,  # List of file prefixes, e.g., ['mse_distribution_A', 'mse_distribution_B', ...]
    t,                 # Number of .npy files per prefix
    names,             # Legend names corresponding to file_prefix_list
    bins=500,          # Number of histogram bins
    colors=None,       # List of colors for each file_prefix
    save_path='mse_distribution_all.pdf'
):
    """
    Reads evaluation result files under multiple file_prefixes, each containing t .npy files,
    computes the average histogram statistics for each prefix, and plots them on the same graph.
    Uses the provided colors for each file_prefix and includes a legend with the provided names.

    Additionally, logs statistical information of log-transformed MSE values.

    :param file_prefix_list: list of str, each element is a file prefix (excluding the _i.npy part)
    :param t: int, number of evaluation runs per file prefix
    :param names: list of str, legend names corresponding to file_prefix_list
    :param bins: int, number of histogram bins
    :param colors: list of str, colors for each file_prefix, e.g., ['#b0a4e3', '#FF5733', ...]
    :param save_path: str, path to save the final plot (PDF format)
    """
    if colors is None:
        # Default colors, reused if not enough colors are provided
        colors = ['#b0a4e3', '#FF5733', '#33FF57', '#3357FF', '#FF33A6']

    # Set Seaborn style
    plt.style.use('seaborn-v0_8-paper')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 2.5))

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for idx, prefix in enumerate(file_prefix_list):
        all_mse_values_list = []
        for i in range(t):
            file_path = os.path.join(prefix, f"mse_distribution_{i}.npy")
            mse_array = np.load(file_path)
            all_mse_values_list.append(mse_array)

        # Concatenate all data for statistical range
        all_concat = np.concatenate(all_mse_values_list)

        log_data = np.log(all_concat)
        print("Log-transformed stats:")
        print(f"Mean: {log_data.mean():.4f}")
        print(f"Std: {log_data.std():.4f}")
        print(f"Skewness: {stats.skew(log_data):.4f}")
        print(f"Kurtosis: {stats.kurtosis(log_data):.4f}")
        min_val, max_val = 0, 0.75
        print(f"[{prefix}] mean: {all_concat.mean():.4f}  max: {all_concat.max():.4f}")

        # Create uniform bin edges
        bins_array = np.linspace(min_val, max_val, bins + 1)

        # Compute and sum histogram counts
        sum_counts = np.zeros(bins)
        for mse_array in all_mse_values_list:
            counts, _ = np.histogram(mse_array, bins=bins_array)
            sum_counts += counts

        # Calculate average counts
        avg_counts = sum_counts / t
        bin_centers = 0.5 * (bins_array[:-1] + bins_array[1:])
        # Apply Gaussian smoothing
        smooth_counts = gaussian_filter1d(avg_counts, sigma=1e-6)

        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'bin_centers': bin_centers,
            'avg_counts': avg_counts
        })
        # Plot the histogram as bars
        ax.bar(
            bin_centers,
            avg_counts,
            width=(bins_array[1] - bins_array[0]),
            color=colors[idx % len(colors)],
            alpha=0.5,
            edgecolor='#000000',
            label=names[idx],
            align='center'
        )

    # Use scientific notation for the y-axis
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
    ax.legend()

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Position the legend in the upper right
    ax.legend(loc='upper right', fontsize=10)

    # Add grid with transparency
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()

    print(f"[Info] Average distribution histogram saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Plot averaged MSE distributions from user-provided .npy results.')
    parser.add_argument('--prefixes', nargs='+', required=True, help='Directories containing mse_distribution_{i}.npy files.')
    parser.add_argument('--names', nargs='+', required=True, help='Legend names, one per prefix.')
    parser.add_argument('--runs', type=int, required=True, help='Number of mse_distribution_{i}.npy files per prefix.')
    parser.add_argument('--bins', type=int, default=500)
    parser.add_argument('--colors', nargs='*', default=None)
    parser.add_argument('--save-path', default='mse_distribution_all.pdf')
    parser.add_argument('--raw-save-path', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if len(args.names) != len(args.prefixes):
        raise ValueError('--names must contain the same number of entries as --prefixes')
    if args.colors is not None and len(args.colors) < len(args.prefixes):
        raise ValueError('--colors must contain at least one color per prefix when provided')

    plot_average_mse_distribution_all_norm(
        file_prefix_list=args.prefixes,
        t=args.runs,
        names=args.names,
        bins=args.bins,
        colors=args.colors,
        save_path=args.save_path,
    )

    if args.raw_save_path:
        plot_average_mse_distribution_all(
            file_prefix_list=args.prefixes,
            t=args.runs,
            names=args.names,
            bins=args.bins,
            colors=args.colors,
            save_path=args.raw_save_path,
        )
