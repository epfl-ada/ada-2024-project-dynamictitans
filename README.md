# Blockbusters and Busts: Unpacking the Gap Between Critics and Cash

## >>>>> [Link to data story](https://minwenmao.github.io/dynamictitans.github.io/) <<<<<

---

## Abstract

The existence of the *Transformers* movie franchise raises a curious question: how can films that critics pan become massive box office successes, while critically acclaimed gems sometimes struggle commercially? This project explores the divide between critical and financial success in cinema - or what makes them over/underperform in general when their critics ratings should predict otherwise - aiming to uncover what drives these differences. We’ll examine correlations between IMDb ratings and box office revenue to reveal general trends and identify outliers. Using a range of methodologies, from time-based and genre analyses to sentiment analysis on plot summaries, we aim to uncover patterns across genres, runtimes, and film attributes. Our approach integrates dataset merging, inflation-adjusted revenue comparisons, outlier classification, distribution analysis, and advanced sentiment modeling to answer our research questions and provide insight into this complex relationship between critics and audiences.

---

## Research Questions

### General Discrepancies:
1. Are there large-scale correlations, or lack thereof, between IMDb ratings and box office revenue across all films?
2. Are there specific outliers that exhibit a significant gap between IMDb ranking and box office revenue?

### Time-Based Trends:
3. Do discrepancies between IMDb ratings and box office revenues show patterns over time?

### Factors of Discrepancy:
4. What characteristics (e.g., length, cast) are common among films with significant discrepancies between IMDb ratings and box office revenue?
5. What "sweet spots" in runtime might each of those outliers occupy, and what do those say about the approach of those films?
6. Can we categorize films as “overperformers” and “underperformers,” or more specifically, as “critically acclaimed box office flops” and “commercially successful critical flops”?
7. Can sentiment analysis of reviews reveal thematic or narrative elements that correlate with these discrepancies or contribute to these categories?

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

1. Dataset Merging and Filtering
- We integrate the CMU and IMDb datasets. Since no direct mapping exists between the two datasets, we merge them based on movie titles.  
- This approach is likely to introduce duplicates, which are filtered out using additional shared identifiers such as runtime and release year.  
- The final dataset retains only movies that match these criteria. To avoid rating bias, we exclude movies with fewer than 30 votes. This yields a refined dataset of 7,440 entries with valid box office revenue records.  

2. Inflation Adjustment
- Box office revenues are adjusted to 2024 USD using the Consumer Price Index (CPI) as a proxy for yearly inflation.  
- U.S. CPI values are used for consistency, given the U.S.’s prominence in global cinema.  
- To corroborate this approach, we also plot the distribution of the dataset by language.

---

### II. General Discrepancy Analysis

1. General Correlation Exploration
- We attempt to identify trends across the entire dataset by using scatterplots and Ordinary Least Squares (OLS) regression to quantify linear relationships.  

2. Correlation Exploration through Specific Cases
- We analyze the correlation between inflation-adjusted box office revenue and IMDb ratings over specific years (1915-2005, in 10-year intervals). 
- Similar correlation analysis is conducted across countries.  

3. Runtime Analysis
- We explore correlations between runtime and both revenue and ratings.  
- This is done using scatterplots and bar charts showing the distribution of runtime, revenue, and the number of movies within various runtime ranges.  
- This serves as groundwork for later analysis on the role of runtime in outlier identification.

---

### III. Outlier Identification

1. Classification of Films by Critical and Commercial Success
- Films are classified based on their inflation-adjusted box office residuals and IMDb ratings:  
  - **“Overperformers”**: Box office residuals ≥ +1 standard deviation from the regression line, and ratings < 6.5.  
  - **“Underperformers”**: Box office residuals ≤ -1 standard deviation from the regression line, and ratings ≥ 6.5.

---

### IV. Distribution Analysis

1. Overall Distribution Analysis
- For high-discrepancy categories (e.g., “overperformers” and “underperformers”), we analyze distributions of attributes such as language and shared cast members.  

2. Runtime Distribution Analysis
- Extending the runtime correlation from Section II, we analyze the distributions of box office revenue and ratings over runtime for both outlier categories, once again using bar charts.  

3. Genre Distribution Analysis
- We apply distribution analysis to the top 10 primary genres within each outlier category.  
- Additionally, we investigate connections between genres and actor gender within each outlier category using tools like chord diagrams.  

---

### V. Sentiment Analysis

We perform sentiment analysis on IMDb user reviews, using two complementary schemes:  

1. Sentiment Distribution
- Sentiments are analyzed using a discrete emotion labeling scheme, employing a BERT model pre-trained on the English Twitter Emotion dataset.  
- This reveals the comparative distribution of emotions like anger and worry across films.  

2. Sentiment Trajectory Plotting
- A continuous negative-to-positive scoring scheme (using the SnowNLP library) is applied to generate sentiment trajectories for each film.  
- Aggregated trajectories form an "average sentiment profile" for each discrepancy type.  
- Further analysis includes:  
  - Distribution of films by the number of notable emotional fluctuations.  
  - Distribution of films by overall magnitude of emotional span.  
  - Distribution of films by emotional trajectory (slopes) and baselines (y-intercepts), derived using linear regression.  


### VI. Current Results Highlight

Our preliminary findings suggest the following:  
- **Runtime Advantage:** Overperforming films tend to perform slightly better with longer runtimes compared to underperforming films.  
- **Emotional Dynamics:** Overperforming films exhibit a higher proportion of "surprise" elements, more emotional twists, and a broader overall emotional range.  
- **Emotional Arc:** Despite their emotional range, overperforming films show overall fewer signs of a genuinely upward-rising emotional arc.

## Future Directions
Further research could explore or eliminate the influence of additional factors on box office revenue, such as the number of screens. Understanding these factors may refine our classification and analysis of both overperforming and underperforming films.

---

## Timeline

- **Nov 15:** Finish II
- **Nov 19:** Finish III
- **Dec 12:** Finish IV, V
- **Dec 17:** Finish data story draft
- **Dec 20:** Finish webpage, Final quality check, adjustments


---

## Team Organization

- **Leo:** Initial obtaining of additional datasets, merging and cleaning data, Task I, help with creating website
- **Ruoxi:** Initial analysis, Task I, Task III (co-completed), Task V, refinement of data story text
- **Gengfu:** Initial analysis, Task I, Task II, Task III (co-completed), refinement of data story text
- **Minwen:** Initial analysis, Task II, Task III (co-completed), Task IV, refinement of data story text, creating and final hosting of website
- **Kaile:** problem formulation, README.md, repository compiling, clean-up, and quality check, data story drafting

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

# run bert.ipynb to finetune the BERT model used for discrete sentiment labelling -> label.ipnyb to obtain the labels
# run summary.ipynb to obtain continuous sentiment scoring

# run/see results.ipynb notebook for current results

# scrape the actor images used for visualization of top actors/actresses in each outlier
python src/script/image_crawler.py


```



### How to use the library
Initial data preprocessing python script from the full CMU and IMDb datasets go in `script/data_preprocessing.py` and is run separately from the notebook. All plotting methods called by the Tasks I, II, III sections of the notebook go in `src/script/initial_analysis_plots_static.py`; all methods called by the Task IV section of the notebook go in `src/script/outlier_analysis_plots.py`; and all methods called by the Task V section of the notebook go in `src/script/sentiment_analysis.py`.

External notebooks required for Task V include: `src/script/bert.ipynb` (for finetuning the BERT model), `src/script/label.ipynb` (for discrete sentiment labelling using the BERT model), and `src/script/summary.ipynb` (for continuous bianry sentiment scoring). Scripts and methods called by the notebook related to tests such as k-means clustering tests employed in the process and any future tests go in `tests/`.


## Project Structure

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory (note that the model we specifically has not been directly included due to Github space constraints)
│   ├── script                         <- Utility scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── requirements.txt        <- File for installing python dependencies
└── README.md
```
