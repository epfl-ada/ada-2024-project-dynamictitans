import os
import pandas as pd
import numpy as np
import json

def get_data_path(file_name):
    """
    Get the full path of a data file.
    :param file_name: The file name, e.g., 'movie.metadata.tsv'
    :return: The full file path
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '../data')
    return os.path.join(data_dir, file_name)

def load_data():
    """
    Load movie, character, and IMDb data from files.
    :return: DataFrames for movie, character, and IMDb data
    """
    movie_file_path = get_data_path('movie.metadata.tsv')
    character_file_path = get_data_path('character.metadata.tsv')
    imdb1_file_path = get_data_path('title.ratings.tsv')
    imdb2_file_path = get_data_path('title.basics.tsv')
    
    movie = pd.read_csv(movie_file_path, delimiter='\t', header=None)
    character = pd.read_csv(character_file_path, delimiter='\t', header=None)
    # imdb1 has ratings, imdb identifier and number of votes
    imdb1 = pd.read_csv(imdb1_file_path, delimiter='\t')
    # imdb2 has genres, title, year, etc
    imdb2 = pd.read_csv(imdb2_file_path, delimiter='\t', low_memory=False)
    
    return movie, character, imdb1, imdb2

def preprocess_movie_data(movie_df):
    """
    Preprocess the movie data.
    :param movie_df: The movie DataFrame
    :return: The preprocessed movie DataFrame
    """
    movie_df.columns = ['Wikipedia movie ID', 'Freebase movie ID', 'Movie name', 'Movie release date', 
                        'Movie box office revenue', 'Movie runtime', 'Movie languages', 'Movie countries', 
                        'Movie genres']
    movie_df["Movie name"] = movie_df["Movie name"].str.capitalize()
    movie_df["Movie name"] = movie_df["Movie name"].str.replace(r'[éè]', 'e', regex=True)
    
    return movie_df

def preprocess_imdb_data(imdb1_df, imdb2_df):
    """
    Preprocess the IMDb data by merging and filtering relevant information.
    :param imdb1_df: IMDb ratings DataFrame
    :param imdb2_df: IMDb basics DataFrame
    :return: The preprocessed IMDb DataFrame
    """
    merged_df = pd.merge(imdb1_df, imdb2_df, on='tconst', how='inner')
    # in the imdb data set they include series and a lot of other things.
    # This removes a lot of movies that are not in the other data 
    merged_2 = merged_df[(merged_df.titleType == 'movie') | (merged_df.titleType == 'tvMovie')]
    # removing all the movies that have less than 30 votes 
    # questionable step because some of the movies in our data set have 
    # very low amount of votes
    merged_3 = merged_2[merged_2['numVotes'] >= 30]
    # making the titles of the movies in the same format for both data set 
    # for example in one data set a movie can be called "The matrix" and in the other "The Matrix"
    merged_3.loc[:, 'primaryTitle'] = merged_3['primaryTitle'].str.capitalize()
    # removing all the columns that are not needed
    merged_4 = merged_3.drop(columns=['numVotes', 'titleType', 'isAdult', 'endYear', 'originalTitle', 'runtimeMinutes', 'genres', 'tconst'])
    
    return merged_4

def merge_movie_imdb_data(movie_df, imdb_df):
    """
    Merge the movie DataFrame with the IMDb DataFrame.
    :param movie_df: The movie DataFrame
    :param imdb_df: The IMDb DataFrame
    :return: The merged DataFrame
    """
    # merging the data sets on our movie data set
    merged_final = pd.merge(imdb_df, movie_df, left_on='primaryTitle', right_on='Movie name', how='right')
    # creating a column that has the year of the movie release so that we can remove duplicates 
    # that have the same name but different release year
    merged_final["Movie release year"] = merged_final["Movie release date"].str[:4]
    merged_final2 = merged_final[merged_final['Movie release year'] == merged_final['startYear']]
    # drop colllumn that are not needed
    merged_final2 = merged_final2.drop(columns=['primaryTitle', 'startYear', 'Movie release date'])
    merged_final2 = merged_final2.drop(merged_final2[merged_final2['Movie box office revenue'].isnull()].index)
    
    merged_final2['startYear'] = pd.to_numeric(merged_final['startYear'], errors='coerce').astype('Int64')
    merged_final2['Movie release year'] = merged_final2['Movie release year'].astype('int')
    merged_final2['Movie genres'] = merged_final2['Movie genres'].apply(lambda x: list(json.loads(x).values()))
    merged_final2['Primary Country'] = merged_final2['Movie countries'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    
    return merged_final2

def preprocess_inflation_data():
    """
    Load and preprocess the inflation data.
    :return: The preprocessed inflation DataFrame
    """
    inflation_file_path = get_data_path('CPI-US-Iflation.xlsx')
    inflation = pd.read_excel(inflation_file_path, header=None)
    inflation.columns = ['Year', 'CPI']
    # Initialize with floats to avoid dtype issues
    inflation['rate'] = 1.0  
    inflation = inflation.drop(inflation[inflation['Year'] > 2020].index)
    
    # Calculate the currency rate
    for i in range(len(inflation)):
        inf = 1
        for j in range(len(inflation) - 1, i, -1):
            inf *= (1 + inflation.loc[j, 'CPI'] / 100.0)
        inflation.loc[i, 'rate'] = inf  # Use .loc to avoid chained assignment
    
    inflation['Year'] = inflation['Year'].astype('int')
    
    return inflation

def enrich_dataset():
    '''
    Use external datset to enrich existing revenue dataset(not completed yet)
    '''
    file_path = get_data_path('TMDB_movie_dataset_v11.csv')

if __name__ == "__main__":
    # Load the data
    movie, character, imdb1, imdb2 = load_data()
    
    # Preprocess the movie data
    movie = preprocess_movie_data(movie)
    
    # Preprocess the IMDb data
    imdb = preprocess_imdb_data(imdb1, imdb2)
    
    # Merge the movie and IMDb data
    merged_data = merge_movie_imdb_data(movie, imdb)
    
    # Print the first few rows of the merged data to verify the results
    print(merged_data.head())
    
    # Save the cleaned data
    output_dir = os.path.join(get_data_path(''), '../../data')
    os.makedirs(output_dir, exist_ok=True)
    merged_data.to_csv(os.path.join(output_dir, 'cleaned_data.csv'), index=False)
    
    # Preprocess the inflation data 
    inflation = preprocess_inflation_data()
    print(inflation.head())
    merged_data.to_csv(os.path.join(output_dir, 'inflation.csv'), index=False)
