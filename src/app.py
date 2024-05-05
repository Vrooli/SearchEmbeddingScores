import os
import numpy as np
import matplotlib.pyplot as plt

# Define the output directory
OUTPUT_DIR = './plots'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def calculate_score(distance, top_metric, sort_by):
    epsilon = 0.01
    base_score = 1 / (distance + epsilon)  # Ensure this does not lead to infinity
    # Adding a debug print to check values received by function
    # print(f"Calculating score with distance: {distance}, top_metric: {top_metric}, base_score: {base_score}")
    if sort_by == 'EmbedTopAsc':
        return base_score - top_metric
    elif sort_by == 'EmbedTopDesc':
        return base_score + top_metric
    elif sort_by.startswith('EmbedDateCreated') or sort_by.startswith('EmbedDateUpdated'):
        return base_score + (top_metric if 'Desc' in sort_by else -top_metric)
    return base_score

def generate_plots(sort_by):
    # Create graphs for different scales (x, y, label)
    scales = [
        ((0, 1), (0, 10), 'XS Range'),
        ((0, 1), (0, 100), 'S Range'),
        ((0, 1), (0, 10000), 'L Range'),
        ((0, 1), (0, 1000000), 'XL Range'),
    ]

    # Arrange output plots vertially
    plot_size = 8
    fig, axes = plt.subplots(len(scales), 1, figsize=(plot_size, plot_size * len(scales)))
    fig.suptitle(f'Scenario: {sort_by}')
    colormap = 'plasma' # Can be 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    
    # Generate graph for each scale
    for ax, scale in zip(axes, scales):
        # Unpack the scale
        x_range, y_range, label = scale
        # Generate points for the graph. 100 points for each axis, for a total of 10000 points
        x_points = np.linspace(*x_range, 100)
        y_points = np.linspace(*y_range, 100)
        # Calculate the score for each point
        scores = np.array([[calculate_score(x, y, sort_by) for x in x_points] for y in y_points])

        im = ax.imshow(scores, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower', aspect='auto', cmap=colormap)
        # Label the plot
        ax.set_title(label)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Bookmarks')
        fig.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout(rect=[0, 0.25, 1, 0.95])
    plt.savefig(f'{OUTPUT_DIR}/{sort_by}_combined.png')
    plt.close()

# Define scenarios
sort_options = [
    'EmbedTopAsc',
    'EmbedTopDesc',
    'EmbedDateCreatedAsc',
    'EmbedDateCreatedDesc'
]

# Generate combined plots for each scenario
for sort_option in sort_options:
    generate_plots(sort_option)

print('Plots generated successfully! Exiting...')
