import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import ast
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from scipy.interpolate import make_interp_spline


def plot_data_languages(cleaned_data):
    # Extract all languages into a list
    languages_list = cleaned_data['Movie languages'].dropna().apply(lambda x: list(ast.literal_eval(x).values()))
    flattened_languages = [lang for sublist in languages_list for lang in sublist]
    language_counts = pd.Series(flattened_languages).value_counts()

    # Select the top 10 most common languages
    top_10_languages = language_counts.head(10)

    # Plot the distribution of the top 10 movie languages
    plt.figure(figsize=(10, 6))
    top_10_languages.plot(kind='bar', alpha=0.75, color='orange')
    plt.title('Top 10 Movie Languages')
    plt.xlabel('Languages')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_runtime_influence(cleaned_data):

    # Analyze the influence of Movie Runtime on averageRating and Adjusted_Revenue
    # Extract runtime, averageRating, and Adjusted_Revenue data
    runtime_data = cleaned_data[['Movie runtime', 'averageRating', 'Adjusted_Revenue']].dropna()

    # Plot Runtime vs Average Rating
    plt.figure(figsize=(10, 6))
    plt.scatter(runtime_data['Movie runtime'], runtime_data['averageRating'], alpha=0.6)
    plt.title('Movie Runtime vs Average Rating', fontsize=20)
    plt.xlabel('Runtime (minutes)', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot Runtime vs Adjusted Revenue
    plt.figure(figsize=(10, 6))
    plt.scatter(runtime_data['Movie runtime'], runtime_data['Adjusted_Revenue'], alpha=0.6)
    plt.title('Movie Runtime vs Adjusted Revenue', fontsize=20)
    plt.xlabel('Runtime (minutes)', fontsize=14)
    plt.ylabel('Adjusted Revenue (scaled)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_runtime_influence_distr_short(cleaned_data):

    # Extract runtime, averageRating, and Adjusted_Revenue data
    runtime_data = cleaned_data[['Movie runtime', 'averageRating', 'Adjusted_Revenue']].dropna()

    # Remove runtime=1003 outlier
    runtime_data = runtime_data[runtime_data['Movie runtime'] != 1003]

    # Create bins: Divide runtime from 0 to 380 into 20-minute intervals
    bins = list(range(0, 381, 20))

    # Calculate the average value of averageRating and Adjusted_Revenue for each bin
    runtime_data['Runtime Bin'] = pd.cut(runtime_data['Movie runtime'], bins=bins, right=False)
    binned_data = runtime_data.groupby('Runtime Bin').agg({
        'averageRating': 'mean',
        'Adjusted_Revenue': 'mean'
    }).reset_index()

    # Remove bins with missing values
    binned_data = binned_data.dropna()

    # Calculate the center point of each bin for smooth curve plotting
    bin_centers = [(interval.left + interval.right) / 2 for interval in binned_data['Runtime Bin']]

    # Smooth Average Rating and Adjusted Revenue
    x_smooth = np.linspace(min(bin_centers), max(bin_centers), 300)
    y_avg_smooth = make_interp_spline(bin_centers, binned_data['averageRating'])(x_smooth)
    y_rev_smooth = make_interp_spline(bin_centers, binned_data['Adjusted_Revenue'])(x_smooth)

    # Plot Runtime vs Average Rating: Bar Chart and Smoothed Line Chart
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, binned_data['averageRating'], width=15, alpha=0.6, color='skyblue', label='Average Rating')
    plt.plot(x_smooth, y_avg_smooth, color='green', linewidth=2, label='Smoothed Line')
    plt.title('Runtime vs Average Rating with Smoothed Line Chart', fontsize=20)
    plt.xlabel('Runtime (minutes)', fontsize=14)
    plt.ylabel('Average Rating', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot Runtime vs Adjusted Revenue: Bar Chart and Smoothed Line Chart
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, binned_data['Adjusted_Revenue'], width=15, alpha=0.6, color='salmon', label='Adjusted Revenue')
    plt.plot(x_smooth, y_rev_smooth, color='green', linewidth=2, label='Smoothed Line')
    plt.title('Runtime vs Adjusted Revenue with Smoothed Line Chart', fontsize=20)
    plt.xlabel('Runtime (minutes)', fontsize=14)
    plt.ylabel('Adjusted Revenue (scaled)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_runtime_influence_distr(cleaned_data, name):

    colors = {"overperformers": "red", "underperformers": "blue", "general": "purple"}
    color = colors[name]

    # Extract runtime, averageRating, and Adjusted_Revenue data
    runtime_data = cleaned_data[['Movie runtime', 'averageRating', 'Adjusted_Revenue']].dropna()

    # Remove runtime=1003 outlier
    runtime_data = runtime_data[runtime_data['Movie runtime'] != 1003]

    # Create bins: Divide runtime from 0 to 380 into 20-minute intervals
    bins = list(range(0, 381, 20))

    # Calculate the average value of averageRating, Adjusted_Revenue, and Movie Count for each bin
    runtime_data['Runtime Bin'] = pd.cut(runtime_data['Movie runtime'], bins=bins, right=False)
    binned_data = runtime_data.groupby('Runtime Bin').agg({
        'averageRating': 'mean',
        'Adjusted_Revenue': 'mean',
        'Movie runtime': 'count'
    }).reset_index()

    binned_data = binned_data.rename(columns={'Movie runtime': 'Movie Count'})

    # Remove bins with missing values
    binned_data = binned_data.dropna()

    # Calculate the center point of each bin for smooth curve plotting
    bin_centers = [(interval.left + interval.right) / 2 for interval in binned_data['Runtime Bin']]

    # Smooth Average Rating and Adjusted Revenue
    x_smooth = np.linspace(min(bin_centers), max(bin_centers), 300)
    y_avg_smooth = make_interp_spline(bin_centers, binned_data['averageRating'])(x_smooth)
    y_rev_smooth = make_interp_spline(bin_centers, binned_data['Adjusted_Revenue'])(x_smooth)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Average Rating (left y-axis)
    ax1.bar(bin_centers, binned_data['averageRating'], width=15, alpha=0.6, color=color, label='Average Rating')
    ax1.plot(x_smooth, y_avg_smooth, color='green', linewidth=2, label='Smoothed Avg Rating')
    ax1.set_xlabel('Runtime (minutes)', fontsize=14)
    ax1.set_ylabel('Average Rating', fontsize=14, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.set_ylim(bottom=0)

    # Plot Movie Count (right y-axis)
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, binned_data['Movie Count'], width=15, alpha=0.4, color='orange', label='Movie Count')
    ax2.set_ylabel('Movie Count', fontsize=14, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.set_ylim(bottom=0)

    plt.title(f'Runtime vs Average Rating and Movie Count, {name}', fontsize=20)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot Runtime vs Adjusted Revenue and Movie Count
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Adjusted Revenue (left y-axis)
    ax1.bar(bin_centers, binned_data['Adjusted_Revenue'], width=15, alpha=0.6, color=color, label='Adjusted Revenue')
    ax1.plot(x_smooth, y_rev_smooth, color='green', linewidth=2, label='Smoothed Adj Revenue')
    ax1.set_xlabel('Runtime (minutes)', fontsize=14)
    ax1.set_ylabel('Adjusted Revenue (scaled)', fontsize=14, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.set_ylim(bottom=0)

    # Plot Movie Count (right y-axis)
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, binned_data['Movie Count'], width=15, alpha=0.4, color='orange', label='Movie Count')
    ax2.set_ylabel('Movie Count', fontsize=14, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.set_ylim(bottom=0)

    plt.title(f'Runtime vs Adjusted Revenue and Movie Count, {name}', fontsize=20)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_countries_revenue_rating(cleaned_data):
    
    # Filter data to ensure necessary columns have valid values
    filtered_data = cleaned_data.dropna(subset=['averageRating', 'Adjusted_Revenue', 'Movie countries'])

    # Parse 'Movie countries' column and retain the first country
    filtered_data['Movie countries'] = filtered_data['Movie countries'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    filtered_data['Primary Country'] = filtered_data['Movie countries'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown')

    # Compute position_density: weighted average of rating and revenue (closer to top-right corner)
    filtered_data['Normalized Rating'] = filtered_data['averageRating'] / filtered_data['averageRating'].max()
    filtered_data['Normalized Revenue'] = filtered_data['Adjusted_Revenue'] / filtered_data['Adjusted_Revenue'].max()
    filtered_data['Position Density'] = (filtered_data['Normalized Rating'] + filtered_data['Normalized Revenue']) / 2

    # Set up the scatter plot
    plt.figure(figsize=(10, 7))
    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.Reds

    # Scatter points with color mapped to Position Density
    scatter = plt.scatter(
        filtered_data['averageRating'],
        filtered_data['Adjusted_Revenue'],
        c=filtered_data['Position Density'],
        cmap=cmap,
        norm=norm,
        s=100,
        alpha=0.8,
        edgecolors='k',
        label=None
    )

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Position Density', fontsize=14)

    # Label axes and add title
    plt.xlabel('Average Rating', fontsize=14)
    plt.ylabel('Adjusted Revenue (USD)', fontsize=14)
    plt.title('Movie Ratings vs Adjusted Revenue with Position-Based Colors', fontsize=16)
    
    # Add grid and improve layout
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.show()


def animate_country_rating_revenue(cleaned_data):
    filtered_data = cleaned_data.dropna(subset=['averageRating', 'Adjusted_Revenue', 'Movie release year', 'Movie countries'])

    # Parse 'Movie countries' and retain the first country
    filtered_data['Movie countries'] = filtered_data['Movie countries'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    filtered_data['Primary Country'] = filtered_data['Movie countries'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown')

    # Ensure the year is of integer type and sort by year in ascending order
    filtered_data['Movie release year'] = filtered_data['Movie release year'].astype(int)
    filtered_data = filtered_data.sort_values(by='Movie release year', ascending=True)

    # Get all combinations of years and countries to ensure full coverage of all countries and years
    all_years = filtered_data['Movie release year'].unique()
    all_countries = filtered_data['Primary Country'].unique()
    all_combinations = pd.MultiIndex.from_product(
        [all_years, all_countries], names=['Movie release year', 'Primary Country']
    ).to_frame(index=False)

    # Merge with the original data and fill missing values
    filled_data = all_combinations.merge(filtered_data, on=['Movie release year', 'Primary Country'], how='left')
    filled_data['averageRating'] = filled_data['averageRating'].fillna(0.1)  # Fill missing ratings with a default value
    filled_data['Adjusted_Revenue'] = filled_data['Adjusted_Revenue'].fillna(0.1)  # Fill missing revenue with a default value

    # Get all unique countries and assign unique colors to each
    unique_countries = filled_data['Primary Country'].unique()
    base_colors = pc.qualitative.Bold + pc.qualitative.D3 + pc.qualitative.Light24  # High-contrast color palettes
    color_map = {country: base_colors[i % len(base_colors)] for i, country in enumerate(unique_countries)}

    # Force chronological animation frames to ensure playback progresses year by year
    filled_data['Primary Country'] = filled_data['Primary Country'].astype(str)

    # Create an animated scatter plot
    fig = px.scatter(
        filled_data,
        x='averageRating',
        y='Adjusted_Revenue',
        animation_frame='Movie release year',
        color='Primary Country',
        title="Movie Ratings and Adjusted Revenue Over Time",
        labels={
            "averageRating": "Average Rating",
            "Adjusted_Revenue": "Adjusted Revenue (USD)",
            "Movie release year": "Release Year",
            "Primary Country": "Country"
        },
        hover_data=['Primary Country'],
        size='Adjusted_Revenue',
        size_max=80,
        color_discrete_map=color_map,
        template='plotly',
        width=1200,
        height=700
    )

    # Ensure all countries appear in the legend
    fig.for_each_trace(lambda t: t.update(marker=dict(opacity=0.7)))

    # Beautify layout
    fig.update_layout(
        xaxis=dict(title="Average Rating", range=[0, 10]),
        yaxis=dict(title="Adjusted Revenue (USD)", range=[0, filled_data['Adjusted_Revenue'].max() * 1.1]),
        legend_title="Primary Country",
        margin=dict(l=50, r=50, t=80, b=50),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                        method="animate",
                        args=[None, {"frame": {"duration": 1000, "redraw": True},
                                    "fromcurrent": True, "mode": "immediate"}]),
                    dict(
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])])
        ]
    )

    fig.show()


def plot_ratio_vs_language(cleaned_data, higher, lower):

    # List of valid languages to filter out non-language categories
    valid_languages = [
        'English', 'French', 'Spanish', 'German', 'Chinese', 'Japanese', 'Italian', 'Korean', 'Russian', 'Hindi',
        'Portuguese', 'Arabic', 'Dutch', 'Swedish', 'Turkish', 'Danish', 'Norwegian', 'Greek', 'Polish',
        'Thai', 'Finnish', 'Hebrew', 'Czech', 'Hungarian', 'Vietnamese', 'Malay', 'Indonesian', 'Romanian',
        'Bulgarian', 'Serbian'
    ]

    # Function to filter Top 30 languages
    def filter_top_languages(df, top_languages):
        return df[df['Movie languages'].apply(lambda x: any(lang in ''.join(list(ast.literal_eval(x).values())) for lang in top_languages))]

    # Clean 'Movie languages' to remove invalid entries
    language_counts = cleaned_data['Movie languages'].apply(lambda x: list(ast.literal_eval(x).values()))
    flattened_languages = [lang for sublist in language_counts for lang in sublist if any(valid in lang for valid in valid_languages)]

    # Extract Top 30 valid movie languages
    top_languages = pd.Series(flattened_languages).value_counts().head(30).index

    # Filter data for Top 30 languages
    higher_top30 = filter_top_languages(higher, top_languages)
    lower_top30 = filter_top_languages(lower, top_languages)
    cleaned_data_top30 = filter_top_languages(cleaned_data, top_languages)

    # Function to calculate performance ratios
    def calculate_language_performance(higher, lower, cleaned_data):
        # Count total movies per language
        language_counts = cleaned_data['Movie languages'].apply(lambda x: list(ast.literal_eval(x).values()))
        total_count = pd.Series([lang for sublist in language_counts for lang in sublist if any(valid in lang for valid in valid_languages)]).value_counts()

        # Count overperformed movies per language
        higher_languages = higher['Movie languages'].apply(lambda x: list(ast.literal_eval(x).values()))
        overperformed_count = pd.Series([lang for sublist in higher_languages for lang in sublist if any(valid in lang for valid in valid_languages)]).value_counts()

        # Count underperformed movies per language
        lower_languages = lower['Movie languages'].apply(lambda x: list(ast.literal_eval(x).values()))
        underperformed_count = pd.Series([lang for sublist in lower_languages for lang in sublist if any(valid in lang for valid in valid_languages)]).value_counts()

        # Calculate proportions
        overperformed_ratio = (overperformed_count / total_count).fillna(0)
        underperformed_ratio = (underperformed_count / total_count).fillna(0)

        return overperformed_ratio, underperformed_ratio

    # Calculate performance ratios for Top 30 languages
    over_ratio_top30, under_ratio_top30 = calculate_language_performance(higher_top30, lower_top30, cleaned_data_top30)

    # Extract data for visualization
    data = pd.DataFrame({
        'Language': top_languages[:30],
        'Overperformed': [over_ratio_top30.get(lang, 0) for lang in top_languages[:30]],
        'Underperformed': [under_ratio_top30.get(lang, 0) for lang in top_languages[:30]]
    })
    
    data['Ratio'] = data['Overperformed'] / data['Underperformed']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set up axes
    x = data['Underperformed']
    y = data['Overperformed']
    z = np.zeros(len(data))  # Base for bars
    dx = dy = 0.005  # Width and depth of bars
    dz = data['Ratio']

    # Plot bars
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(data)))

    for i in range(len(data)):
        ax.bar3d(
            x[i], y[i], z[i], dx, dy, dz[i],
            color=colors[i], edgecolor='black', label=data['Language'][i] if i < 10 else ""
        )
        ax.text(
            x[i], y[i], dz[i] + 0.2,
            data['Language'][i],
            ha='center', va='bottom', fontsize=10, rotation=45
        )

    # Set labels and title
    ax.set_xlabel('Underperformed', fontsize=12)
    ax.set_ylabel('Overperformed', fontsize=12)
    ax.set_zlabel('Ratio (Overperformed/Underperformed)', fontsize=12)
    ax.set_title('Top 30 Movie Languages: Overperformed vs Underperformed (3D)', fontsize=16)

    # Adjust layout and show
    ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.2, 1.0), title='Languages')
    plt.tight_layout()
    plt.show()