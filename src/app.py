# Creates graphs to visualize the scoring function for different sorting options
# Sort options include:
# - EmbedTopAsc: High-bookmarked items weighted lower
# - EmbedTopDesc: High-bookmarked items weighted higher
# - EmbedDateCreatedAsc: Items created farther from target date weighted higher
# - EmbedDateCreatedDesc: Items created closer to target date weighted higher
# - EmbedDateUpdatedAsc and EmbedDateUpdatedDesc: Similar to EmbedDateCreated but based on updated date, so 
#   no need to test separately
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import SymLogNorm
import datetime

# Define the output directory
OUTPUT_DIR = './plots'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calculate_top_score(distance, bookmarks, is_descending):
    """
    Calculate the score based on the given distance, bookmarks, and order.

    Parameters:
    distance (float): The euclidean distance between the search query's embedding and the item's embedding. From 0 to 1.
    bookmarks (int): The number of bookmarks for the item.
    is_descending (bool): If true, higher bookmark counts are considered better.

    Returns:
    float: The calculated score.

    Raises:
    None
    """
    epsilon = 0.01 # This ensures that the division does not lead to infinity
    distance_factor = 1 / ((distance + epsilon) ** 2)  # Exponent to increase the impact of distance
    if is_descending:
        # (1 / (POWER(distance + 0.01, 2))) + (LN(bookmarks + 1) * 2)
        result = distance_factor + np.log1p(bookmarks) * 2
    else:
        # (1 / (POWER(distance + 0.01, 2))) + (EXP(-bookmarks / 5) * 20)
        result = distance_factor + np.exp(-bookmarks / 5) * 20
    return result
    
def calculate_date_score(distance, item_date, reference_date, is_descending):
    """
    Calculate the score based on the distance, the item's date, the reference date, and order.

    Parameters:
    distance (float): The euclidean distance between the search query's embedding and the item's embedding.
    item_date (datetime.datetime): The date associated with the item.
    reference_date (datetime.datetime): The reference date for comparison.
    is_descending (bool): If true, dates closer to the reference date are considered better.

    Returns:
    float: The calculated score.

    Raises:
    None
    """
    epsilon = 0.01
    base_score = 1 / (distance + epsilon)
    date_difference = abs((item_date - reference_date).total_seconds()) / 3600  # seconds difference normalized to hours

    if is_descending:
        # (1 / (POWER(distance + 0.01, 2))) + (EXP(-POWER(ABS(EXTRACT(EPOCH FROM item_date - reference_date) / 3600), 0.25)) * 100)
        date_score = np.exp(-abs(date_difference) ** 0.25) * 100
    else:
        # (1 / (POWER(distance + 0.01, 2))) + (LOG(1 + POWER(ABS(EXTRACT(EPOCH FROM item_date - reference_date) / 3600), 0.25)) * 1)
        date_score = np.log1p(abs(date_difference) ** 0.25) * 1

    return base_score + date_score

def generate_plots(scenario, scales, colormap='plasma'):
    """
    Generate plots for different scenarios based on provided scales and a scoring function.

    Parameters:
    scenario (str): A descriptive name for the scenario being plotted.
    scales (list of tuples): Each tuple contains x_range (tuple), y_range (tuple), and a label (str).
    calculate_score (function): The scoring function to be used.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    colormap (str): Color map for the plot.

    Returns:
    None
    """
    plot_size = 8
    fig, axes = plt.subplots(len(scales), 1, figsize=(plot_size, plot_size * len(scales)))
    fig.suptitle(f'Scenario: {scenario}')

    is_descending = 'Desc' in scenario

    for ax, scale in zip(axes, scales):
        x_range, y_range, label = scale
        x_points = np.linspace(*x_range, 100)
        
        # Generate y_points based on whether they are date or numeric
        if isinstance(y_range[0], datetime.timedelta):
            # Handle date scales by converting to total seconds
            ref_date = datetime.datetime.now()
            y_start = (ref_date + y_range[0]).timestamp()
            y_end = (ref_date + y_range[1]).timestamp()
            y_points = [ref_date + datetime.timedelta(seconds=sec) for sec in np.linspace(y_range[0].total_seconds(), y_range[1].total_seconds(), 100)]
            scores = np.array([[calculate_date_score(x, y, ref_date, is_descending) for x in x_points] for y in y_points])
            ylabel = 'Date'
            extent = (x_range[0], x_range[1], y_start, y_end)
        else:
            # Handle numeric scales
            y_points = np.linspace(*y_range, 100)
            scores = np.array([[calculate_top_score(x, y, is_descending) for x in x_points] for y in y_points])
            ylabel = 'Bookmarks'
            extent = (x_range[0], x_range[1], y_range[0], y_range[1])

        # Use a logarithmic color scale, which handles negative values
        norm = SymLogNorm(linthresh=0.01, linscale=1, base=10, vmin=scores.min(), vmax=scores.max())

        im = ax.imshow(scores, extent=extent, origin='lower', aspect='auto', cmap=colormap, norm=norm)
        ax.set_title(label)
        ax.set_xlabel('Distance')
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f'{OUTPUT_DIR}/{scenario}_combined.png')
    plt.close()

# Scales for bookmark-based graphs (x, y, label)
SCALES_BOOKMARKS = [
    ((0, 1), (0, 10), 'XS Range'),
    ((0, 1), (0, 100), 'S Range'),
    ((0, 1), (0, 10000), 'L Range'),
    ((0, 1), (0, 1000000), 'XL Range'),
]

# Scales for date-based graphs (x, y, label)
SCALES_DATES = [
    ((0, 1), (datetime.timedelta(seconds=-1), datetime.timedelta(seconds=1)), 'Seconds Range'),
    ((0, 1), (datetime.timedelta(hours=-1), datetime.timedelta(hours=1)), 'Hours Range'),
    ((0, 1), (datetime.timedelta(days=-1), datetime.timedelta(days=1)), 'Days Range'),
    ((0, 1), (datetime.timedelta(weeks=-1), datetime.timedelta(weeks=1)), 'Weeks Range'),
    ((0, 1), (datetime.timedelta(weeks=-4), datetime.timedelta(weeks=4)), 'Months Range'),
    ((0, 1), (datetime.timedelta(weeks=-52), datetime.timedelta(weeks=52)), 'Years Range'),
]

# Define scenarios
sort_options = [
    'EmbedTopAsc', # Expect graphs to favor bottom-left
    'EmbedTopDesc', # Expect graphs to favor top-left
    'EmbedDateCreatedAsc', # Expect graphs to favor top-left and bottom-left
    'EmbedDateCreatedDesc'  # Expect graphs to favor middle-left
]

# Generate combined plots for each scenario
generate_plots(sort_options[0], SCALES_BOOKMARKS)
generate_plots(sort_options[1], SCALES_BOOKMARKS)
generate_plots(sort_options[2], SCALES_DATES)
generate_plots(sort_options[3], SCALES_DATES)

print('Plots generated successfully! Exiting...')
