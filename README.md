# Blockbusters and Busts: Unpacking the Gap Between Critics and Cash

## Abstract
(insert actually good abstract here)

## Research Questions

### General Discrepancies
- Can we observe any large-scale discrepancies between IMDb ratings and box office revenue across all films?
- Are there specific outliers that exhibit a significant gap between IMDb ranking and box office revenue?

### Genre-Specific Analysis
- How do discrepancies between critical reception and box office revenue vary across different film genres?
- Do certain genres show a stronger correlation between critical acclaim and box office success, or vice versa?

### Time-Based Trends
- Do discrepancies between IMDb ratings and box office revenues exhibit patterns over time?
- Are there specific eras where films were more likely to succeed critically but fail commercially (or the opposite)?

### Factors of Discrepancy
- What characteristics (e.g., length, director, cast) are common among films with a significant discrepancy between IMDb ratings and box office revenue?
- Can we categorize those films as “critically-acclaimed box office flops” or “commercially successful critical flops”?
- Can sentiment analysis of reviews reveal thematic or narrative elements that tend to correlate with these discrepancies or contribute to those two categories?

## Methods

### Data Collection & Integration
- Utilize the IMDb dataset to gather ratings and critical reception metrics for a large set of films.
- Use an additional dataset (e.g., CMU or other box office records) to obtain box office revenue information, ensuring data completeness.

### Ranking and Discrepancy Analysis
- Rank films based on IMDb ratings and box office revenues across all genres and create a discrepancy metric, such as the difference between revenue rank and IMDb rating rank.
- Perform time-period analyses by dividing films into chronological intervals. These intervals could be either equal-length periods or segments with an equal number of movies.

### Genre-Specific Analysis
- Divide films by genre and apply the discrepancy analysis to each genre. This step will help identify genre-specific trends in critical versus commercial success.

### Outlier Identification and Characteristic Aggregation
- Define thresholds for "critically acclaimed" (e.g., IMDb rating ≥ 7.0) and "commercial success" (to be determined by box office revenue distribution).
- Identify films that meet criteria for "critically acclaimed box office flops" or "commercially successful critical flops" and analyze common attributes like length, director, and cast for each type.

### Sentiment Analysis
- Conduct sentiment analysis on IMDb user reviews for each film using natural language processing (NLP) models, analyzing the sentiment trajectories and comparing average sentiment for each discrepancy type. This step will help construct an "average sentiment profile" for critical and commercial outliers.