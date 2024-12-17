import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import statsmodels.api as sm
import pandas as pd

import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import matplotlib.cm as cm
import re

from tslearn.utils import to_time_series_dataset
from scipy.interpolate import interp1d
from collections import Counter
from scipy.signal import find_peaks


def sentiment_distr_piechart(target_movies, label_mapping, target_label="Most Frequent Label", title="overperforming"):
    target_movies[target_label] = target_movies[target_label].replace(label_mapping)
    label_counts = target_movies[target_label].value_counts()
    cmap = cm.get_cmap('Accent', len(label_counts)) 
    color_list = [cmap(i) for i in range(len(label_counts))]
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=color_list)
    plt.title(f"{target_label} Distribution, {title}")
    plt.show()


def plot_sentiment_label_distribution(input_df):
    df = input_df.head(80)

    def extract_numbers(labels):
        numbers = re.findall(r'\d+', labels)  
        return list(map(int, numbers)) 

    df["Extracted Labels"] = df["Predicted Label"].apply(extract_numbers)

    label_distribution = df["Extracted Labels"].apply(lambda x: pd.Series(x).value_counts()).fillna(0)
    label_distribution.index = df["Wikipedia movie ID"]

    ax = label_distribution.plot(
        kind="barh", 
        stacked=True,
        figsize=(12, 6),
        alpha=0.85,
        cmap="RdYlBu"  
    )

    ax.set_title("Predicted Label Distribution (Overperformed)", fontsize=16, pad=15)
    ax.set_xlabel("Count of Predicted Labels", fontsize=12)
    ax.set_yticks([]) 
    ax.set_ylabel("Wikipedia Movie ID", fontsize=12)
    plt.xticks(fontsize=10)

    plt.legend(
        title="Predicted Labels",
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=10,
        title_fontsize=12,
        frameon=True
    )

    plt.tight_layout()

    plt.show()


def standardize_score(senvalue):
    df = senvalue  # Example DataFrame
    time_series_data = df.drop(columns=["Wikipedia movie ID"])

    avg_len = int(np.mean([len(row.dropna()) for _, row in time_series_data.iterrows()]))

    interpolated_series = []
    for _, row in time_series_data.iterrows():
        row = row.dropna().values  # Remove NaN values from the current row
        x_original = np.linspace(0, 1, len(row))  # Original time points
        x_new = np.linspace(0, 1, avg_len)  # New time points based on average length
        interpolator = interp1d(x_original, row, kind='linear', fill_value="extrapolate")
        interpolated_series.append(interpolator(x_new))  # Interpolate the series to the new length

    # Format the interpolated series into the required format for tslearn
    formatted_series = to_time_series_dataset(interpolated_series)
    return formatted_series[:,:,0]


def plot_time_series_kmeans(n, labels, model, h_formatted_series):
    label_counts = Counter(labels)

    for cluster in range(n):
        plt.figure(figsize=(10, 6))  

        for i, label in enumerate(labels):
            if label == cluster:
                plt.plot(h_formatted_series[i].ravel(), color='gray', alpha=0.5)  # 无标签灰色曲线

        plt.plot(
            model.cluster_centers_[cluster].ravel(),
            color='red',
            linewidth=3,
            #linestyle='--',
            label="Centroid"
        )

        num_movies = label_counts[cluster]
        
        plt.title(f"Cluster {cluster} (Movies: {num_movies})", fontsize=16)
        plt.xlabel("Time Steps", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.grid(alpha=0.3) 
        plt.legend(loc="upper right", fontsize=12)
        plt.tight_layout()
        plt.show()



def count_emotional_fluctuations(emotion_curve, threshold=0.0):
    """
    Counts the number of emotional fluctuations (alternating peaks and valleys) in an emotion curve.
    :param emotion_curve: The emotion curve (1D array)
    :param threshold: Fluctuation threshold (used to filter out insignificant fluctuations)
    :return: Number of fluctuations
    """
    peaks, _ = find_peaks(emotion_curve)  # Find peaks in the emotion curve
    valleys, _ = find_peaks(-emotion_curve)  # Find valleys by detecting peaks in the inverted curve
    extrema = np.sort(np.concatenate((peaks, valleys)))  # Combine and sort peaks and valleys
    fluctuation_count = 0
    for i in range(1, len(extrema)):
        # Check if the difference between consecutive extrema exceeds the threshold
        if abs(emotion_curve[extrema[i]] - emotion_curve[extrema[i - 1]]) > threshold:
            fluctuation_count += 1
    return fluctuation_count

def plot_emotional_fluctuation_distr(h_standard_score, l_standard_score): 

    # Step 2: Set the threshold for detecting significant emotional fluctuations
    amplitude_threshold = 0.2  # Define a threshold for significant fluctuations

    # Step 3: Calculate the number of emotional fluctuations for each curve
    # For high-sensitivity group
    h_fluctuations = [
        count_emotional_fluctuations(curve, amplitude_threshold) 
        for curve in h_standard_score
    ]

    # For low-sensitivity group
    l_fluctuations = [
        count_emotional_fluctuations(curve, amplitude_threshold) 
        for curve in l_standard_score
    ]

    # Step 4: Plot histograms for the emotional fluctuations
    plt.figure(figsize=(10, 6))  # Set the overall figure size

    # Plot the histogram for the high-sensitivity group's emotional fluctuations
    plt.subplot(2, 1, 1)  # Create a subplot in the top panel
    plt.hist(h_fluctuations, bins=15, color='blue', alpha=0.7, edgecolor='black')  # Enhanced visual properties
    plt.title("Emotional Fluctuations in Over-Performed Group")  # Add a title
    plt.xlabel("Number of Fluctuations")  # Label the x-axis
    plt.ylabel("Frequency")  # Label the y-axis

    # Plot the histogram for the low-sensitivity group's emotional fluctuations
    plt.subplot(2, 1, 2)  # Create a subplot in the bottom panel
    plt.hist(l_fluctuations, bins=15, color='green', alpha=0.7, edgecolor='black')  # Enhanced visual properties
    plt.title("Emotional Fluctuations in Under-Performed Group")  # Add a title
    plt.xlabel("Number of Fluctuations")  # Label the x-axis
    plt.ylabel("Frequency")  # Label the y-axis

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout()

    # Display the histograms
    plt.show()



def count_emotion_magnitude(emotion_curve, threshold=0.0):
    """
    Calculates the magnitude of emotional fluctuations in an emotion curve.
    The magnitude is defined as the difference between the maximum and minimum values in the curve.

    :param emotion_curve: The emotion curve (1D array or list of values).
    :param threshold: Threshold (not used in the current implementation, but kept for potential extensions).
    :return: The magnitude of fluctuations (float).
    """
    magnitude = np.max(emotion_curve) - np.min(emotion_curve)  # Compute the range of values (max - min)
    return magnitude  # Return the magnitude


def plot_emotional_magnitude_distr(h_standard_score, l_standard_score):

    # Set the amplitude threshold for filtering emotional magnitudes
    amplitude_threshold = 0.2

    # Calculate the magnitude of emotional fluctuations for each curve in the high-sensitivity group
    h_fluctuations = [count_emotion_magnitude(curve, amplitude_threshold) for curve in h_standard_score]

    # Calculate the magnitude of emotional fluctuations for each curve in the low-sensitivity group
    l_fluctuations = [count_emotion_magnitude(curve, amplitude_threshold) for curve in l_standard_score]

    # Plot the histogram for the high-sensitivity group's emotional fluctuation magnitudes
    plt.subplot(2, 1, 1)  # Create a subplot (top panel)
    plt.hist(h_fluctuations, bins=15, edgecolor='black')  # Add edgecolor for better visibility
    plt.xlim(0, 1)  # Set the x-axis limits for consistency
    plt.title("High-Sensitivity Group: Emotional Magnitude")  # Add a title
    plt.xlabel("Magnitude")  # Label the x-axis
    plt.ylabel("Frequency")  # Label the y-axis

    # Plot the histogram for the low-sensitivity group's emotional fluctuation magnitudes
    plt.subplot(2, 1, 2)  # Create a subplot (bottom panel)
    plt.hist(l_fluctuations, bins=15, edgecolor='black')  # Add edgecolor for better visibility
    plt.xlim(0, 1)  # Set the x-axis limits for consistency
    plt.title("Low-Sensitivity Group: Emotional Magnitude")  # Add a title
    plt.xlabel("Magnitude")  # Label the x-axis
    plt.ylabel("Frequency")  # Label the y-axis

    # Adjust layout for better visualization
    plt.tight_layout()

    # Display the plots
    plt.show()


def linear_regression_on_sequences(standard):
    """
    Performs linear regression on a set of sequences and filters results based on the p-value of the slope.
    
    :param standard: List or array of sequences (each sequence is a list or 1D array).
    :return: DataFrame containing the regression results for sequences with significant slopes.
    """
    results = []
    for idx, sequence in enumerate(standard):
        # Convert the sequence to a NumPy array
        y = np.array(sequence)  # Dependent variable (y)
        x = np.arange(len(y))   # Independent variable (x), a sequence of integers from 0 to len(y)-1
        
        # Add a constant term (intercept) to the independent variable
        x = sm.add_constant(x)
        
        # Fit the ordinary least squares (OLS) regression model
        model = sm.OLS(y, x).fit()
        
        # Extract regression coefficients and p-values
        coefficients = model.params  # Regression coefficients [intercept, slope]
        p_values = model.pvalues     # p-values for the coefficients [intercept, slope]
        
        # Check if the p-value for the slope is below the significance threshold (0.1)
        if p_values[1] < 0.1:  # Consider slope significant only if p-value < 0.1
            # Save the results for this sequence
            results.append({
                "sequence_index": idx,            # Sequence index
                "intercept": coefficients[0],     # Intercept
                "slope": coefficients[1],         # Slope
                "p_value_intercept": p_values[0], # p-value for the intercept
                "p_value_slope": p_values[1]      # p-value for the slope
            })
    
    # Return the results as a pandas DataFrame for better usability and analysis
    return pd.DataFrame(results)