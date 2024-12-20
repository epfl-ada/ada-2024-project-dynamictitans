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


def plot_sentiment_label_distribution(input_df, label_mapping):
    df = input_df.head(80)

    def extract_numbers(labels):
        numbers = re.findall(r'\d+', labels)  
        return list(map(int, numbers)) 

    df["Extracted Labels"] = df["Predicted Label"].apply(extract_numbers)

    label_distribution = df["Extracted Labels"].apply(lambda x: pd.Series(x).value_counts()).fillna(0)
    label_distribution.index = df["Wikipedia movie ID"]
    columns_to_map = {key: label_mapping[key] for key in label_distribution.columns if key in label_mapping}
    label_distribution = label_distribution.rename(columns=columns_to_map)

    ax = label_distribution.plot(
        kind="barh", 
        stacked=True,
        figsize=(12, 6),
        alpha=0.85,
        cmap="RdYlBu"  
    )

    ax.set_title("Predicted Label Distribution", fontsize=16, pad=15)
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
                plt.plot(h_formatted_series[i].ravel(), color='gray', alpha=0.5)  # All the other trajectories marked in gray

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

def plot_time_series_mean(labels, model, h_formatted_series, name):
    label_counts = Counter(labels)

    plt.figure(figsize=(10, 6))  

    for i, label in enumerate(labels):
        if label == 0:
            plt.plot(h_formatted_series[i].ravel(), color='gray', alpha=0.5)  # All the other trajectories marked in gray

    plt.plot(
        model.cluster_centers_[0].ravel(),
        color='red',
        linewidth=3,
        #linestyle='--',
        label="Centroid"
    )

    num_movies = label_counts[0]
        
    plt.title(f"{name} (Movies: {num_movies})", fontsize=16)
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

    # Set the threshold for detecting significant emotional fluctuations
    amplitude_threshold = 0.2 

    # Calculate the number of emotional fluctuations for each category
    h_fluctuations = [
        count_emotional_fluctuations(curve, amplitude_threshold) 
        for curve in h_standard_score
    ]

    l_fluctuations = [
        count_emotional_fluctuations(curve, amplitude_threshold) 
        for curve in l_standard_score
    ]

    plt.figure(figsize=(10, 6))

    # Plot the histogram for overperformers' emotional fluctuations
    plt.subplot(2, 1, 1)
    plt.hist(h_fluctuations, bins=15, color='red', alpha=0.7, edgecolor='black')
    plt.title("Emotional Fluctuations in Overperformers")
    plt.xlabel("Number of Fluctuations")
    plt.ylabel("Frequency")

    # Ditto for underperformers
    plt.subplot(2, 1, 2)
    plt.hist(l_fluctuations, bins=15, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Emotional Fluctuations in Underperformers")
    plt.xlabel("Number of Fluctuations")
    plt.ylabel("Frequency")

    plt.tight_layout()
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

    # Calculate the magnitude of emotional fluctuations for each curve among the overperformers
    h_fluctuations = [count_emotion_magnitude(curve, amplitude_threshold) for curve in h_standard_score]

    # Ditto for the underperformers
    l_fluctuations = [count_emotion_magnitude(curve, amplitude_threshold) for curve in l_standard_score]

    # Plot the histogram for overperformers' emotional fluctuation magnitudes
    plt.subplot(2, 1, 1)
    plt.hist(h_fluctuations, bins=15, color='red', edgecolor='black')
    plt.xlim(0, 1)
    plt.title("Overperformers: Emotional Magnitude")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")

    # Plot the histogram for underperformers' emotional fluctuation magnitudes
    plt.subplot(2, 1, 2)
    plt.hist(l_fluctuations, bins=15, color='blue', edgecolor='black')
    plt.xlim(0, 1)
    plt.title("Underperformers: Emotional Magnitude")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def linear_regression_on_sequences(standard):
    """
    Performs linear regression on a set of sequences and filters results based on the p-value of the slope.
    
    :param standard: List or array of sequences (each sequence is a list or 1D array).
    :return: DataFrame containing the regression results for sequences with significant slopes.
    """
    results = []
    for idx, sequence in enumerate(standard):
        y = np.array(sequence)
        x = np.arange(len(y))   # a sequence of integers from 0 to len(y)-1
        
        # Add a constant term/intercept to the independent variable
        x = sm.add_constant(x)
    
        model = sm.OLS(y, x).fit()
        
        # Extract regression coefficients and p-values
        coefficients = model.params 
        p_values = model.pvalues 
        
        # We use a significance threshold of 0.1 here
        if p_values[1] < 0.1:
            results.append({
                "sequence_index": idx,     
                "intercept": coefficients[0], 
                "slope": coefficients[1], 
                "p_value_intercept": p_values[0], 
                "p_value_slope": p_values[1]
            })
    
    return pd.DataFrame(results)

def most_frequent(labels):
    if labels:  # Ensure the list is not empty
        numbers = re.findall(r'\d+', labels)  # Extract all numeric labels in a list, for example ['3', '3', '3', '3', '0', '4']
        numbers = list(map(int, numbers)) 
        return Counter(numbers).most_common(1)[0][0]  # Get the most frequent element
    return None  # Return None if the list is empty

def second_most_frequent(labels):
    if labels:  # Ensure the list is not empty
        numbers = re.findall(r'\d+', labels)  # Extract all numeric labels in a list, for example ['3', '3', '3', '3', '0', '4']
        numbers = list(map(int, numbers)) 
        counts = Counter(numbers).most_common()  # Rank by frequency
        # Return SECOND most frequent if there are indeed two or more emotions, else return the most frequent
        if len(counts) > 1:
            return counts[1][0]
        elif counts:
            return counts[0][0]
    return None  # Return None if the list is empty


def plot_regression_slope_distr(h_regression_results, l_regression_results):
    # Plot histogram of slopes for overperformers
    plt.figure(figsize=(8,6))
    plt.subplot(2, 2, 1) 
    plt.hist(h_regression_results['slope'], bins=15, color='red', edgecolor='black')
    plt.xlim(-0.01, 0.01)
    plt.title("Overperformers: Slope Distribution")
    plt.xlabel("Slope")
    plt.ylabel("Frequency")

    # Plot histogram of slopes for underperformers
    plt.subplot(2, 2, 2)
    plt.hist(l_regression_results['slope'], bins=15, color='blue', edgecolor='black')
    plt.xlim(-0.03, 0.03)
    plt.title("Underperformers: Slope Distribution")
    plt.xlabel("Slope")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_regression_intercept_distr(h_regression_results, l_regression_results):
    # Plot histogram of intercepts for overperformers
    plt.figure(figsize=(8,6))
    plt.subplot(2, 2, 1)
    plt.hist(h_regression_results['intercept'], bins=15, color='red', edgecolor='black')
    plt.title("Overperformers: Intercept Distribution")
    plt.xlabel("Intercept")
    plt.ylabel("Frequency")

    # Plot histogram of intercepts for underperformers
    plt.subplot(2, 2, 2) 
    plt.hist(l_regression_results['intercept'], bins=15, color='blue', edgecolor='black')
    plt.title("Underperformers: Intercept Distribution")
    plt.xlabel("Intercept")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()