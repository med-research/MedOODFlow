import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from openood.utils.vis_comm import save_fig_and_close


def plot_bar_from_csv(file_path, output_dir, ylim, rotate, fixed_bar_width,
                      show_avg):
    # Load data
    data = pd.read_csv(file_path)

    # Forward-fill category names for easier handling
    if 'Category' in data.columns:
        data['Category'] = data['Category'].ffill()

    # Identify metrics (all columns except Category and Sub-Category)
    non_metric_cols = ['Category', 'Sub-Category']
    metrics = [col for col in data.columns if col not in non_metric_cols]

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Define categories
    if 'Category' in data.columns:
        categories = data['Category'].unique()

        # Plot data category-wise, each in a separate figure
        for category in categories:
            subset = data[data['Category'] == category]
            _create_bar_plot(subset, metrics, category, file_name, output_dir,
                             ylim, rotate, fixed_bar_width, show_avg)
    else:
        # If no category column, plot everything in one figure
        _create_bar_plot(data, metrics, 'Results', file_name, output_dir, ylim,
                         rotate, fixed_bar_width, show_avg)


def _create_bar_plot(data, metrics, title, file_name, output_dir, ylim, rotate,
                     fixed_bar_width, show_avg):
    sub_categories = data['Sub-Category'].values
    num_sub_categories = len(sub_categories)
    num_metrics = len(metrics)

    # Fixed parameters
    margin_inches = 1.0  # Left/right margins in inches
    legend_space_inches = (-3, 3)[show_avg]  # space for legend + avg labels
    left_padding_inches = fixed_bar_width * 0.85  # Pad before leftmost group

    # Calculate number of valid bars for each sub-category
    valid_bars_per_group = data[metrics].notna().sum(axis=1).values

    # Calculate actual width needed for the bars
    total_bars = valid_bars_per_group.sum()
    bars_width_inches = total_bars * fixed_bar_width

    # Add space between groups - each gap is the same width as a bar
    # We need (num_sub_categories - 1) gaps between sub-category groups
    gaps_width_inches = (num_sub_categories - 1) * fixed_bar_width \
        if num_sub_categories > 1 else 0

    # Total width needed for all data (bars + gaps + left padding)
    total_data_width_inches = bars_width_inches + \
        gaps_width_inches + left_padding_inches

    # Calculate figure width needed to achieve the fixed bar width
    fig_width = total_data_width_inches + \
        2 * margin_inches + legend_space_inches

    # Fixed figure height with extra space for rotated labels
    fig_height = 3 if rotate == 90 else 2

    # Create figure with calculated dimensions
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Calculate axes width ratio
    axes_width = (total_data_width_inches / fig_width)
    left_margin_ratio = margin_inches / fig_width

    # Create main plot area with space reserved for labels
    bottom_margin = 0.25 if rotate == 90 else 0.15
    ax = fig.add_axes([left_margin_ratio, bottom_margin, axes_width, 0.7])

    # Bar width in data units
    bar_width_data = 1.0

    # Calculate left padding in data units
    padding_data_units = (left_padding_inches / fixed_bar_width)

    # Calculate group positions dynamically
    group_widths_data = valid_bars_per_group * bar_width_data
    group_positions = np.zeros(num_sub_categories)
    group_centers = np.zeros(num_sub_categories)
    current_pos = padding_data_units
    for i in range(num_sub_categories):
        group_positions[i] = current_pos
        group_centers[i] = current_pos + group_widths_data[i] / 2
        current_pos += group_widths_data[i] + bar_width_data + \
            left_padding_inches  # Add gap

    # Find maximum height for padding calculation
    max_value = data[metrics].max().max() if not data[metrics].empty else 0

    # Store average values for each metric
    avg_values = {}

    for metric in metrics:
        # Calculate and store average for this metric, ignoring NaNs
        if show_avg:
            avg_values[metric] = data[metric].mean() \
                if data[metric].notna().any() else 0

    # Get color cycle from matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Calculate the y-axis limit
    if not ylim:
        ylim = max_value

    # Add 15% padding to the top
    display_max = ylim * 1.15

    for i, metric in enumerate(metrics):
        metric_color = colors[i % len(colors)]
        # Plot bars for each sub-category for the current metric if not NaN
        for j, sub_cat in enumerate(sub_categories):
            value = data.iloc[j][metric]
            if pd.notna(value):
                # Find how many valid bars are before this one in the group
                valid_metrics_before = data.iloc[j][metrics[:i]].notna().sum()
                bar_pos = group_positions[j] + \
                    valid_metrics_before * bar_width_data - \
                    padding_data_units / 2
                bar = ax.bar(bar_pos,
                             value,
                             bar_width_data,
                             alpha=0.6,
                             color=metric_color,
                             align='edge')

                # Add value labels on top of each bar
                height = bar[0].get_height()
                value_str = f'{value:.0f}' \
                    if np.abs(value - round(value)) < 0.01 else f'{value:.1f}'
                if value_str.endswith('.0'):
                    value_str = value_str[:-2]
                ax.text(bar[0].get_x() + bar[0].get_width() / 2.,
                        height + 0.5,
                        value_str,
                        ha='center',
                        va='bottom',
                        rotation=rotate,
                        fontsize=8,
                        fontweight='bold')

    # Create a legend for metrics
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i % len(colors)], alpha=0.6)
        for i in range(num_metrics)
    ]
    # Replace the legend positioning code with this cleaner solution
    # Calculate the right edge of the plot in figure coordinates
    fig_right_edge = left_margin_ratio + axes_width
    legend_x = fig_right_edge + 0.05  # Fixed 10% of figure width as padding
    # Position the legend using figure coordinates instead of axes coordinates
    ax.legend(handles=legend_handles,
              labels=metrics,
              loc='center left',
              bbox_to_anchor=(legend_x, 0.6),
              bbox_transform=fig.transFigure)

    if show_avg:
        # Draw average lines and labels
        total_data_units = current_pos - bar_width_data \
            if num_sub_categories > 0 else 0
        for i, metric in enumerate(metrics):
            if metric in avg_values:
                avg = avg_values[metric]
                metric_color = colors[i % len(colors)]
                avg_str = f'{avg:.2f}'
                if avg_str.endswith('.00'):
                    avg_str = avg_str[:-3]

                # Draw horizontal dashed line
                ax.hlines(avg,
                          0,
                          total_data_units,
                          colors=metric_color,
                          linestyles='dashed',
                          alpha=0.4,
                          linewidth=1.5)

                # Calculate vertical offset for the label to avoid collisions
                vertical_offset = 0
                for prev_i in range(i):
                    if metrics[prev_i] in avg_values:
                        prev_avg = avg_values[metrics[prev_i]]
                        if abs(avg - prev_avg) < (display_max * 0.03):
                            vertical_offset = (i % 3 - 1) * \
                                              (display_max * 0.04)

                # Add average value label on the right side
                ax.text(total_data_units + padding_data_units / 2,
                        avg + vertical_offset,
                        avg_str,
                        va='center',
                        ha='left',
                        color=metric_color,
                        fontsize=8,
                        fontweight='bold')

    # Set x-ticks at group centers
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(group_centers)
    ax.set_xticklabels(sub_categories,
                       rotation=rotate,
                       ha='right' if rotate == 90 else 'center',
                       fontsize=10)
    ax.set_ylabel('Metric Value', fontsize=12)

    # Set x limits to ensure all bars are visible with padding
    x_min = 0  # Start from 0 to include the left padding
    total_data_units = current_pos - bar_width_data \
        if num_sub_categories > 0 else 0
    ax.set_xlim(x_min, total_data_units)

    # Add alternating background for better readability
    for i in range(len(sub_categories)):
        if i % 2 == 1:
            ax.axvspan(group_positions[i] - bar_width_data,
                       group_positions[i] + group_widths_data[i] +
                       left_padding_inches,
                       alpha=0.1,
                       color='gray')

    # Set up y-axis
    ax.set_ylim(0, display_max)
    step = ylim / 5
    yticks = np.arange(0, ylim + step, step)
    yticks = yticks[yticks <= ylim]
    ax.set_yticks(yticks)

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    safe_title = title.replace(' ', '_').replace('/', '_')
    output_path = os.path.join(output_dir, f'{file_name}_{safe_title}.svg')
    save_fig_and_close(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create bar plots from a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('output_dir',
                        type=str,
                        help='Directory to save the output plot images.')
    parser.add_argument('--ylim',
                        type=float,
                        default=None,
                        help='Upper limit for y-axis. Default: auto')
    parser.add_argument('--rotate',
                        type=int,
                        default=90,
                        help='Rotate labels by this angle. Default: 90')
    parser.add_argument('--fixed_bar_width',
                        type=float,
                        default=0.15,
                        help='Fixed width of each bar. Default: 0.15')
    parser.add_argument('--show-avg',
                        action='store_true',
                        help='Show average lines and labels.')

    args = parser.parse_args()

    # Call the plotting function with the provided arguments
    plot_bar_from_csv(args.file_path, args.output_dir, args.ylim, args.rotate,
                      args.fixed_bar_width, args.show_avg)
