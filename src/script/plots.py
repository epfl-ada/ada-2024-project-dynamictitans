import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import pandas as pd

def plot_by_year_nonadjusted(cleaned_data):
    """
    Plots box office revenue not adjusted for inflation vs year.
    :param cleaned_data: Processed dataframe
    :return: Line plot
    """
    # Calculate the average box office revenue per release year
    box_office_by_year = cleaned_data.groupby('Movie release year')['Movie box office revenue'].mean().reset_index()

    # Set up the figure size for the plot
    plt.figure(figsize=(10, 6))

    # Plot the average box office revenue by release year, with markers on each point
    plt.plot(box_office_by_year['Movie release year'], box_office_by_year['Movie box office revenue'], marker='o')

    # Label the x and y axes
    plt.xlabel('Year')
    plt.ylabel('Average Box Office Revenue')

    # Set the title for the plot
    plt.title('Yearly Variation in Average Box Office Revenue')

    # Add a grid for better readability
    plt.grid(True)

    # Display the plot
    plt.show()


def plot_by_year_comparison_outlier(cleaned_data, outlier_years):
    """
    Plots box office revenue ADJUSTED for inflation vs year and also compared with unadjusted data. Use only after the adjust for inflation part in the notebook.
    :param cleaned_data: Processed dataframe
    :param outlier_years: Top years to highlight, 0 to disable
    :return: Line plot
    """
    # Group by year and calculate the sum of both revenues per year
    revenue_by_year = cleaned_data.groupby('Movie release year')[['Adjusted_Revenue', 'Movie box office revenue']].mean().reset_index()
    years = revenue_by_year['Movie release year']
    adjusted_revenue = revenue_by_year['Adjusted_Revenue']
    if outlier_years != 0:
        top_revenue = revenue_by_year.sort_values(by='Adjusted_Revenue', ascending=False).head(outlier_years)
        top_years = top_revenue['Movie release year'].to_list()

    # Plot the line chart
    plt.figure(figsize=(12, 6))
    plt.plot(revenue_by_year['Movie release year'], revenue_by_year['Adjusted_Revenue'], label='Adjusted Revenue', color='blue', marker='o')
    plt.plot(revenue_by_year['Movie release year'], revenue_by_year['Movie box office revenue'], label='Original Box Office Revenue', color='orange', marker='o')
    if outlier_years != 0: 
        for year in top_years:
            if year in years.values:  # Check if the year is in the data
                revenue = adjusted_revenue[years == year].values[0]  # Get revenue value for the year
                plt.scatter(year, revenue, s=150, edgecolor='darkred', facecolor='none', linewidth=2)

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Revenue')
    plt.title('Yearly Comparison of Adjusted Revenue and Original Box Office Revenue')
    plt.legend()
    plt.grid(True)

    # Display the chart
    plt.show()


def plot_distr_unadjusted(cleaned_data):
    """
    Plots the side-by-side hexbin joint plot plus kernel density estimate plot of box office revenue adjusted for inflation directly vs IMDb rating
    :param cleaned_data: Processed dataframe
    :return: hexbin + kde plot
    """

    # Set the Seaborn theme style to white grid and use a muted color palette
    sns.set_theme(style="whitegrid", palette="muted")

    # Create a joint plot with Seaborn, with a hexbin plot as the main part of the chart
    g = sns.jointplot(data=cleaned_data, x='Adjusted_Revenue', y='averageRating', kind="hex", cmap="Blues", height=8, gridsize=50, marginal_kws=dict(color="dodgerblue"))

    # Overlay a density contour plot in the center area of the joint plot to show distribution contours
    sns.kdeplot(data=cleaned_data, x='Adjusted_Revenue', y='averageRating', cmap="Blues", fill=True, alpha=0.4, ax=g.ax_joint)

    # Set the margins of the chart to fill the light blue background
    g.ax_joint.margins(0)  
    g.ax_joint.set_xlim(cleaned_data['Adjusted_Revenue'].min(), cleaned_data['Adjusted_Revenue'].max())
    g.ax_joint.set_ylim(cleaned_data['averageRating'].min(), cleaned_data['averageRating'].max())

    # Set the title and labels of the plot and adjust the plot
    g.fig.suptitle("Distribution of Worldwide Gross vs IMDb Rating", fontsize=16, weight='bold', ha='center')
    g.set_axis_labels("Worldwide Gross (in billions)", "IMDb Rating", fontsize=12)
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)
    plt.show()


def plot_distr_adjusted(cleaned_data):
    """
    Plots the side-by-side hexbin joint plot plus kernel density estimate plot of box office revenue adjusted for inflation and (1+log)'d vs IMDb rating
    :param cleaned_data: Processed dataframe
    :return: hexbin + kde plot
    """
    # Set Seaborn theme for the plot (white grid and muted color palette)
    sns.set_theme(style="whitegrid", palette="muted")

    # Create a hexbin jointplot to show the relationship between Adjusted World Gross (Log_Revenue) and IMDb Rating
    g = sns.jointplot(data=cleaned_data, x='Log_Revenue', y='averageRating', kind="hex", cmap="Blues", height=8,gridsize=60, marginal_kws=dict(color="dodgerblue"))

    # Overlay a KDE (Kernel Density Estimate) plot on the jointplot to show the density distribution
    sns.kdeplot(data=cleaned_data, x='Log_Revenue', y='averageRating', cmap="Blues", fill=True, alpha=0.4, ax=g.ax_joint)

    # Remove margins on the plot for a tighter fit
    g.ax_joint.margins(0) 

    # Set the x-axis and y-axis limits based on the data range
    g.ax_joint.set_xlim(cleaned_data['Log_Revenue'].min(), cleaned_data['Log_Revenue'].max())
    g.ax_joint.set_ylim(cleaned_data['averageRating'].min(), cleaned_data['averageRating'].max())

    # Set the title for the plot and adjust font size and position
    g.fig.suptitle("Distribution of Worldwide Gross (Adjusted) vs IMDb Rating", fontsize=16, weight='bold', ha='center')

    # Set the labels for the x and y axes with appropriate font size
    g.set_axis_labels("Worldwide Gross (Adjusted, in billions)", "IMDb Rating", fontsize=12)

    # Adjust the layout to make it more compact and prevent label overlap
    g.fig.tight_layout()

    # Adjust the subplot layout to ensure the title does not overlap with the plot
    g.fig.subplots_adjust(top=0.95)

    # Display the plot
    plt.show()


def plot_correlation_general(cleaned_data, model):
    """
    Plots the log revenue vs rating scatterplot of all movie data, calculates R2 correlation score, while also plotting +1 standard deviation interval around the linear regression line
    :return: scatterplot
    """
    # Calculate the coefficients, predictions, and standard deviation
    coefficients = model.params
    predictions = model.predict()
    std_dev = np.std(predictions - cleaned_data['Log_Revenue'])
    r2_score = model.rsquared_adj

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(cleaned_data['averageRating'], cleaned_data['Log_Revenue'], 
                color='#3498DB', alpha=0.5, s=50, label='Data Points')

    # Regression line
    plt.plot(cleaned_data['averageRating'], predictions, color='orange', linewidth=2, label='Regression Line')

    # Standard deviation lines
    plt.plot(cleaned_data['averageRating'], predictions + std_dev, color='green', linestyle='--', linewidth=2, label='+1 Std Dev')
    plt.plot(cleaned_data['averageRating'], predictions - std_dev, color='red', linestyle='--', linewidth=2, label='-1 Std Dev')

    # Add labels and title
    plt.xlabel('Average Rating', fontsize=12)
    plt.ylabel('Log of Movie Box Office Revenue', fontsize=12)
    plt.title(f'Linear Regression of Log Box Office Revenue on Average Rating (R² = {r2_score:.2f})', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_correlation_per_country(cleaned_data):
    """
    Plots the log revenue vs rating scatterplot of movie data per select countries, calculates R2 correlation score, while also plotting +1 standard deviation interval around the linear regression line
    :param cleaned_data: Processed dataframe
    :return: scatterplot
    """
    # Select the top 9 countries with the highest movie counts and get their names in a list
    selected_countries = cleaned_data['Primary Country'].value_counts().sort_values(ascending=False).head(9).index.tolist()

    # Filter the data to include only the selected countries
    subset = cleaned_data[cleaned_data['Primary Country'].isin(selected_countries)]

    # Generate a color palette for each country based on the number of selected countries
    colors_countries = plt.cm.tab10(np.linspace(0, 1, len(selected_countries)))

    # Set up the figure size for the plot
    plt.figure(figsize=(18, 12))

    # Loop through each selected country and create a subplot for each
    for i, country in enumerate(selected_countries, 1):
        plt.subplot(3, 3, i)  
        country_data = subset[subset['Primary Country'] == country]  

        # Perform OLS regression for Log_Revenue based on averageRating for the current country
        model_countries = smf.ols(formula='Log_Revenue ~ averageRating', data=country_data).fit(cov_type='HC2')
        coefficients_countries = model_countries.params  
        predictions_countries = model_countries.predict()  
        r2_score_countries = model_countries.rsquared_adj  

        # Calculate the standard deviation of residuals for the regression model
        std_dev_countries = np.std(predictions_countries - country_data['Log_Revenue'])

        # Plot actual data points with unique color for each country
        plt.scatter(country_data['averageRating'], country_data['Log_Revenue'], color=colors_countries[i - 1], alpha=0.6, s=40, edgecolor='white', label='Actual Data')

        # Plot the regression line for the model
        plt.plot(country_data['averageRating'], predictions_countries, color='orange', label='Regression Line')

        # Plot lines for +1 and -1 standard deviations from the regression line
        plt.plot(country_data['averageRating'], predictions_countries + std_dev_countries, color='green', linestyle='--', label='+1 Std Dev')
        plt.plot(country_data['averageRating'], predictions_countries - std_dev_countries, color='red', linestyle='--', label='-1 Std Dev')

        # Set labels and title for the subplot
        plt.xlabel('Average Rating')
        plt.ylabel('Log of Movie Box Office Revenue')
        plt.title(f'Regression: {country} (R² = {r2_score_countries:.2f})')

        # Add grid lines for better readability
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add legend only for the first subplot
        if i == 1:  
            plt.legend(fontsize=9, loc='upper left', frameon=True, framealpha=0.8, edgecolor='gray')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the entire set of subplots
    plt.show()


def plot_correlation_per_timeframe(cleaned_data, selected_years):
    """
    Plots the log revenue vs rating scatterplot of movie data per the selected timeframe specified, calculates R2 correlation score, while also plotting +1 standard deviation interval around the linear regression line
    :param cleaned_data: Processed dataframe
    :param selected_years: Selected timeframe in list or range() format
    :return: scatterplot
    """
    # Find the maximum year in the dataset for reference
    cleaned_data['Year'].max()

    # Subset the data to only include rows where the year is in the selected years
    subset = cleaned_data[cleaned_data['Year'].isin(selected_years)]

    # Generate a color palette for each year based on the number of selected years
    colors_years = plt.cm.tab10(np.linspace(0, 1, len(selected_years)))

    # Set up the figure size for the plot
    plt.figure(figsize=(15, 10))

    # Loop through each selected year and create a subplot for each
    for i, year in enumerate(selected_years, 1):
        plt.subplot(2, 5, i)  
        year_data = subset[subset['Year'] == year] 

        # Perform OLS regression for Log_Revenue based on averageRating for the current year
        model_years = smf.ols(formula='Log_Revenue ~ averageRating', data=year_data).fit(cov_type='HC2')
        coefficients_years = model_years.params  
        predictions_years = model_years.predict()  
        r2_score_years = model_years.rsquared_adj  

        # Calculate the standard deviation of residuals for the regression model
        std_dev_years = np.std(predictions_years - year_data['Log_Revenue'])

        # Plot actual data points with unique color for each year
        plt.scatter(year_data['averageRating'], year_data['Log_Revenue'], color=colors_years[i - 1], alpha=0.6, s=40, edgecolor='white', label='Actual Data')

        # Plot the regression line for the model
        plt.plot(year_data['averageRating'], predictions_years, color='orange', label='Regression Line')

        # Plot lines for +1 and -1 standard deviations from the regression line
        plt.plot(year_data['averageRating'], predictions_years + std_dev_years, color='green', linestyle='--', label='+1 Std Dev')
        plt.plot(year_data['averageRating'], predictions_years - std_dev_years, color='red', linestyle='--', label='-1 Std Dev')

        # Set labels and title for the subplot
        plt.xlabel('Average Rating')
        plt.ylabel('Log of Movie Box Office Revenue')
        plt.title(f'Regression: {year} (R² = {r2_score_years:.2f})')

        # Add grid lines for better readability
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add legend only for the first subplot
        if i == 1:
            plt.legend(fontsize=9, loc='upper left', frameon=True, framealpha=0.8, edgecolor='gray')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the entire set of subplots
    plt.show()


def plot_genre_distribution(cleaned_data, model):
    """
    Plots the barplot of genre distribution of "underperformers" and "overperformers" respectively compared to linear regression data
    :param cleaned_data: Processed dataframe
    :param model: statsmodels.formula.api.ols regression model already fitted on data
    :return: Barplot
    """
    predictions = model.predict()
    std_dev = np.std(predictions - cleaned_data['Log_Revenue'])

    # Calculate residuals between actual and predicted values
    residuals = cleaned_data['Log_Revenue'] - predictions

    # Identify outliers where the absolute residual is greater than the standard deviation
    outliers = cleaned_data[np.abs(residuals) > std_dev]['Log_Revenue']

    # Separate data into high residuals (above +std_dev) and low residuals (below -std_dev)
    higher = cleaned_data[residuals > std_dev].copy()
    lower = cleaned_data[residuals < -std_dev].copy()

    # Extract the first, second, and third genres for each movie in 'higher'
    higher.loc[:, 'First genre'] = higher['Movie genres'].apply(lambda x: x[0] if len(x) > 0 else None)  # First genre
    higher.loc[:, 'Second genre'] = higher['Movie genres'].apply(lambda x: x[1] if len(x) > 1 else None)  # Second genre, if available
    higher.loc[:, 'Third genre'] = higher['Movie genres'].apply(lambda x: x[2] if len(x) > 2 else None)  # Third genre, if available

    # Extract the first, second, and third genres for each movie in 'lower'
    lower.loc[:, 'First genre'] = lower['Movie genres'].apply(lambda x: x[0] if len(x) > 0 else None)  # First genre
    lower.loc[:, 'Second genre'] = lower['Movie genres'].apply(lambda x: x[1] if len(x) > 1 else None)  # Second genre, if available
    lower.loc[:, 'Third genre'] = lower['Movie genres'].apply(lambda x: x[2] if len(x) > 2 else None)  # Third genre, if available

    # Count occurrences of each genre in the first, second, and third positions in 'higher'
    First_t10_h = higher['First genre'].value_counts()   # Top 10 first genres in higher residuals
    Second_t10_h = higher['Second genre'].value_counts()  # Top 10 second genres in higher residuals
    Third_t10_h = higher['Third genre'].value_counts()    # Top 10 third genres in higher residuals

    # Count occurrences of each genre in the first, second, and third positions in 'lower'
    First_t10_l = lower['First genre'].value_counts()   # Top 10 first genres in lower residuals
    Second_t10_l = lower['Second genre'].value_counts()  # Top 10 second genres in lower residuals
    Third_t10_l = lower['Third genre'].value_counts()    # Top 10 third genres in lower residuals

    # Combine all genres (first, second, and third positions) for an overall top 10 in 'higher'
    all_genres_h = pd.concat([higher['First genre'], higher['Second genre'], higher['Third genre']])
    All_t10_h = all_genres_h.value_counts().head(10)  # Top 10 genres overall for higher residuals

    # Combine all genres (first, second, and third positions) for an overall top 10 in 'lower'
    all_genres_l = pd.concat([lower['First genre'], lower['Second genre'], lower['Third genre']])
    All_t10_l = all_genres_l.value_counts().head(10)  # Top 10 genres overall for lower residuals

    # Initialize an array to store genre counts by position in 'higher'
    data_h = np.zeros((len(All_t10_h.index), 3))  # Array with rows as genres and 3 columns for positions
    i = 0

    # Populate 'data_h' with genre counts from the top 10 genres in each position in 'higher'
    for genre in All_t10_h.index:
        data_h[i, 0] = First_t10_h.get(genre, 0)  # Count in first genre position
        data_h[i, 1] = Second_t10_h.get(genre, 0)  # Count in second genre position
        data_h[i, 2] = Third_t10_h.get(genre, 0)   # Count in third genre position
        i += 1

    # Initialize an array to store genre counts by position in 'lower'
    data_l = np.zeros((len(All_t10_l.index), 3))  # Array with rows as genres and 3 columns for positions
    i = 0

    # Populate 'data_l' with genre counts from the top 10 genres in each position in 'lower'
    for genre in All_t10_l.index:
        data_l[i, 0] = First_t10_l.get(genre, 0)  # Count in first genre position
        data_l[i, 1] = Second_t10_l.get(genre, 0)  # Count in second genre position
        data_l[i, 2] = Third_t10_l.get(genre, 0)   # Count in third genre position
        i += 1
    
    categories_h = All_t10_h.index
    subcategories_h = ['First_Genre', 'Second_Genre', 'Third_Genre']
    categories_l = All_t10_l.index
    subcategories_l = ['First_Genre', 'Second_Genre', 'Third_Genre']

    # Set bar width and index offset
    bar_width = 0.5
    index_h = np.arange(len(categories_h))

    # Set figure size
    plt.figure(figsize=(16, 6))

    # Plot data for each subcategory in a stacked bar chart
    for i in range(len(subcategories_h)):
        plt.bar(index_h, data_h[:, i], bar_width, label=subcategories_h[i], bottom=np.sum(data_h[:, :i], axis=1))

    # Add labels and title
    plt.xlabel('Genre Category')
    plt.ylabel('Count')
    plt.title('Top 10 Genres in Overperformed movies')
    plt.xticks(index_h, categories_h)
    plt.legend()

    # Display the plot
    plt.show()

    index_l = np.arange(len(categories_l))

    # Set figure size
    plt.figure(figsize=(16, 6))

    # Plot data for each subcategory in a stacked bar chart
    for i in range(len(subcategories_l)):
        plt.bar(index_l, data_l[:, i], bar_width, label=subcategories_l[i], bottom=np.sum(data_l[:, :i], axis=1))

    # Add labels and title
    plt.xlabel('Genre Category')
    plt.ylabel('Count')
    plt.title('Top 10 Genres in Underperformed movies')
    plt.xticks(index_l, categories_l)
    plt.legend()

    # Display the plot
    plt.show()