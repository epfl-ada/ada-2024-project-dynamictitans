# Blockbusters and Busts: Unpacking the Gap Between Critics and Cash

## Abstract

The existence of the *Transformers* movie franchise raises a curious question: how can films that critics pan become massive box office successes, while critically acclaimed gems sometimes struggle commercially? This project explores the divide between critical and financial success in cinema - or what makes them over/underperform in general when their critics ratings should predict otherwise - aiming to uncover what drives these differences. We’ll examine correlations between IMDb ratings and box office revenue to reveal general trends and identify outliers. Using a range of methodologies, from time-based and genre-specific analyses to sentiment analysis on plot summaries, we aim to uncover patterns across genres, timeframes, and film attributes. Our approach integrates dataset merging, inflation-adjusted revenue comparisons, outlier classification, distribution analysis, and advanced sentiment modeling to answer our research questions and provide insight into this complex relationship between critics and audiences.

---

## Research Questions

### General Discrepancies:
1. Are there large-scale correlations, or lack thereof, between IMDb ratings and box office revenue across all films?
2. Are there specific outliers that exhibit a significant gap between IMDb ranking and box office revenue?

### Genre-Specific Analysis:
3. How do discrepancies between critical reception and box office revenue vary across different film genres?
4. Do certain genres, countries, or other factors show a stronger correlation between critical acclaim and box office success, or vice versa?

### Time-Based Trends:
5. Do discrepancies between IMDb ratings and box office revenues show patterns over time?
6. Are there specific eras when films were more likely to succeed critically but fail commercially (or vice versa)?

### Factors of Discrepancy:
7. What characteristics (e.g., length, director, cast) are common among films with significant discrepancies between IMDb ratings and box office revenue?
8. Can we categorize films as “overperformers” and “underperformers,” or more specifically, as “critically acclaimed box office flops” and “commercially successful critical flops”?
9. Can sentiment analysis of reviews reveal thematic or narrative elements that correlate with these discrepancies or contribute to these categories?

---

## Additional Datasets

- **IMDb Non-commercial Dataset:** 1,497,169 entries covering movies, TV episodes, and short films. This dataset includes:
  - Average IMDb rating (0-10 scale)
  - Genres
  - Title type (e.g., movie, TV episode)
  - Title
  - Runtime

---

## Methods

### I. Initial Data Collection & Integration

1. **Dataset Merging and Filtering:** We will integrate the CMU and IMDb datasets. Since no direct mapping exists between the IMDB dataset and our dataset, we merge the two based on movie titles. This approach is likely to introduce duplicates, which we will then filter out by using additional shared identifiers, such as runtime and release year. Our final dataset will retain only the movies that match these criteria. To avoid rating bias, we’ll filter out movies with fewer than 30 votes, yielding a refined dataset of 7,440 entries with valid box office revenue records.
2. **Inflation Adjustment:** Box office revenues will be adjusted to 2024 USD using the Consumer Price Index (CPI) as a proxy for yearly inflation. We assume U.S. CPI values for consistency, given the U.S.’s prominence in global cinema.

### II. General Discrepancy Analysis

1. **Correlation Exploration by Timeframe:** We will analyze correlations between inflation-adjusted box office revenue and IMDb ratings over specific years (1915-2005, every 10 years) to identify trends and deviations. Scatterplots and Ordinary Least Squares regression will quantify linear relationships.
2. **Correlation Exploration by Genre/Country:** Similar analysis will be conducted across genres and countries, focusing on each movie’s primary genre as listed in the CMU dataset.

### III. Outlier Identification

1. **Classification of Films by Critical and Commercial Success:** Regression analysis will classify films with inflation-adjusted box office residuals of +1 standard deviation or higher as “overperformers” and those at -1 or lower as “underperformers.” A nine-category threshold method was initially considered (e.g., high/medium/low box office and IMDb ratings) but was discarded due to potential arbitrariness. This approach may be revisited if time allows.

### IV. Distribution Analysis

1. **Overall Distribution Analysis:** For high-discrepancy categories (e.g., “overperformers” and “underperformers”), we will analyze distributions of attributes like length, director, and cast, visualizing these with bar charts and confidence intervals. Chi-square or t-tests will assess statistically significant differences.
2. **Genre Distribution Analysis:** The same distribution analysis will be applied to the top 10 primary genres within each discrepancy category.

### V. Sentiment Analysis

1. **Sentiment Trajectory Modeling:** We’ll perform sentiment analysis on IMDb user reviews, examining sentiment per sentence using NLP models like RoBERTa. Sentiment trajectories for each film will be aggregated to form an “average sentiment profile” for each discrepancy type (e.g., "overperformers", or if time allows, even more specific categories such as "critical success but commercial flop") over the review timeline.

---

## Proposed Timeline

- **Nov 15:** Finish II (Minimum Milestone P2 objectives here)
- **Nov 19:** Finish III
- **Dec 9:** Finish IV, V
- **Dec 15:** Finish data story webpage
- **Dec 20:** Final quality check, adjustments

---

## Team Organization

- **Leo:** Initial obtaining of additional datasets, merging and cleaning data, Task I
- **Ruoxi:** Initial analysis, Task I, Task III
- **Gengfu:** Initial analysis, Task I, Task II
- **Minwen:** Initial analysis, Task II, Task III
- **Kaile:** problem formulation, README.md, task V, quality check

## Repository

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-dynamictitans/tree/main
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>

# install requirements
pip install -r requirements.txt

# separately download and copy CMU movie and IMDb datasets (link see results.ipynb notebook)
cp movie.metadata.tsv character.metadata.tsv title.ratings.tsv title.basics.tsv <project repo>/src/data

# preprocess dataset separately
python src/script/data_preprocessing.py

# run/see results.ipynb notebook for current results


```



### How to use the library
Initial data preprocessing python script from the full CMU and IMDb datasets go in `script/data_preprocessing.py` and is run separately from the notebook. All plotting methods called by the notebook go in `src/script/plots.py`. Scripts and methods called by the notebook related to tests such as k-means clustering tests employed in the process and any future tests go in `tests/`.


## Project Structure

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── script                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── requirements.txt        <- File for installing python dependencies
└── README.md
```
