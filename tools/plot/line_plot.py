import argparse
import pandas as pd
import matplotlib.pyplot as plt
from openood.utils.vis_comm import save_fig_and_close, MARKERS


def plot_csv_data(file_path, title, output_file, alpha, break_axis=None):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Extract the column names
    columns = data.columns

    # Calculate legend rows to determine figure height
    ncol = 3
    num_legend_items = len(columns) - 1
    num_legend_rows = (num_legend_items + ncol - 1) // ncol

    first_col_name = columns[0]

    # Separate style rows from data
    style_labels = ['linestyle', 'marker', 'color']
    style_rows = data[data[first_col_name].isin(style_labels)]
    data_rows = data[~data[first_col_name].isin(style_labels)].copy()

    # Convert data columns to numeric, turn non-numerics into NaN
    for col in data_rows.columns[1:]:
        data_rows[col] = pd.to_numeric(data_rows[col], errors='coerce')

    # X-axis values (first column as strings)
    x_labels = data_rows[first_col_name].astype(str)
    # Use the index for evenly spaced x-ticks
    x = range(len(x_labels))

    if break_axis:
        # Parse break_axis: "zoom_min,zoom_max[,ratio]"
        parts = [float(p) for p in break_axis.split(',')]
        zoom_min, zoom_max = parts[:2]
        ratio = parts[2] if len(parts) > 2 else 1.0

        # Determine the full data range
        all_data = data_rows.iloc[:, 1:].values.flatten()
        all_data = all_data[~pd.isna(all_data)]
        data_min = min(all_data) if len(all_data) > 0 else 0
        data_max = max(all_data) if len(all_data) > 0 else 1

        # Decide which part is upper and which is lower
        overlap_height = 1.5  # REMOVE IF NEEDED
        if zoom_max < data_max:
            # Zoomed part is lower, main part is upper
            y1_min, y1_max = zoom_min, zoom_max
            y2_min, y2_max = zoom_max - overlap_height, data_max
            height_ratios = [1, ratio]
        else:
            # Zoomed part is upper, main part is lower
            y1_min, y1_max = data_min, zoom_min - overlap_height
            y2_min, y2_max = zoom_min, zoom_max
            height_ratios = [ratio, 1]

        fig_height = 4 + num_legend_rows * 0.2
        fig, (ax_upper, ax_lower) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(6, fig_height),
            gridspec_kw={'height_ratios': height_ratios})
        axes = [ax_upper, ax_lower]

        # Add margin to top and bottom of the combined plot
        margin_upper = (y2_max - y2_min) * 0.1
        margin_lower = (y1_max - y1_min) * 0.1
        ax_upper.set_ylim(y2_min, y2_max + margin_upper)
        ax_lower.set_ylim(y1_min - margin_lower, y1_max)

        ax_upper.spines['bottom'].set_visible(False)
        ax_lower.spines['top'].set_visible(False)
        ax_upper.xaxis.tick_top()
        ax_upper.tick_params(labeltop=False)  # don't put x-axis labels on top
        ax_lower.xaxis.tick_bottom()

        # Add break marks
        d = .015  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False)
        ax_upper.plot((-d, +d), (-d, +d), **kwargs)
        ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_lower.transAxes)
        ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    else:
        fig_height = 4 + num_legend_rows * 0.2
        fig, ax = plt.subplots(figsize=(6, fig_height))
        axes = [ax]
        # Add margin to y-axis
        ax.margins(y=0.05)

    # Plot each column from the second onward
    marker_adj = 0
    for i, column in enumerate(columns[1:]):
        plot_kwargs = {
            'marker': MARKERS[(i + marker_adj) % len(MARKERS)],
            'label': 'Average' if i == 0 else column,
            'alpha': alpha
        }

        # Override with styles from CSV if they exist
        for style in style_labels:
            style_value = style_rows.loc[style_rows[first_col_name] == style,
                                         column]
            if not style_value.empty and pd.notna(style_value.iloc[0]):
                # Use .iloc[0] to get the scalar value
                plot_kwargs[style] = style_value.iloc[0]
                # Adjust marker index if marker is specified
                marker_adj -= int(style == 'marker')

        for ax in axes:
            ax.plot(x, data_rows[column], **plot_kwargs)
            ax.grid(True)

    # Set plot title and axis labels
    axes[0].set_title(title)
    if break_axis:
        ax_lower.set_xlabel(columns[0])
        # Calculate the vertical center between the two subplots
        # Get the positions of the subplots
        pos_upper = ax_upper.get_position()
        pos_lower = ax_lower.get_position()
        # Calculate the center y position
        y_center = (pos_lower.y0 + pos_upper.y1) / 2 + 0.1
        fig.text(0.02, y_center, columns[1], va='center', rotation='vertical')
    else:
        axes[0].set_xlabel(columns[0])
        axes[0].set_ylabel(columns[1])

    # Set x-axis ticks to evenly spaced values with labels from column 1
    plt.xticks(x, x_labels)

    # Add legend outside the plot
    if break_axis:
        ax_lower.legend(loc='upper center',
                        bbox_to_anchor=(0.5, -0.25),
                        ncol=ncol)
    else:
        axes[0].legend(loc='upper center',
                       bbox_to_anchor=(0.5, -0.15),
                       ncol=ncol)

    # Add grid for better visibility
    # plt.grid(True) # This is now inside the loop for each axis

    # Adjust the layout to make room for the legend and avoid overlaps
    plt.tight_layout()
    if break_axis:
        fig.subplots_adjust(hspace=0.05)

    # Save the plot to the specified output file
    save_fig_and_close(output_file)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Plot a CSV file and save the output as an image file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('title', type=str, help='Title of the plot.')
    parser.add_argument('output_file',
                        type=str,
                        help='Path to save the output plot image.')
    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help='Alpha transparency for the plot lines.')
    parser.add_argument('--break-axis',
                        type=str,
                        default=None,
                        help='Create a broken y-axis to zoom into a region. '
                        'Format: "zoom_min,zoom_max[,ratio]". '
                        'Example: "99,100,3" creates a magnified view '
                        'of the [99, 100] range, making it 3x taller '
                        'than the rest of the plot.')

    args = parser.parse_args()

    # Call the plotting function with the provided arguments
    plot_csv_data(args.file_path, args.title, args.output_file, args.alpha,
                  args.break_axis)
