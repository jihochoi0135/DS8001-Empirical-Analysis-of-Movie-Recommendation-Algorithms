import os
import pandas as pd

#os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest-small")
os.chdir(r"C:/Users/Jason/OneDrive/TMU/Fall 25/DS8001/Project/ml-latest")

ratings = pd.read_csv("ratings.csv")
movies  = pd.read_csv("movies.csv")
tags    = pd.read_csv("tags.csv")
links   = pd.read_csv("links.csv")

#Glimpse
print(ratings.info())
print(movies.info())
print(tags.info())
print(links.info())

# Convert timestamps to datetime
ratings["rating_datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
tags["tag_datetime"] = pd.to_datetime(tags["timestamp"], unit="s")

# Check na
print(ratings.isna().sum())
print(movies.isna().sum())
print(tags.isna().sum())
print(links.isna().sum())

# Left join ratings and movies by movieId
ratings_movies = ratings.merge(movies, on="movieId", how="left")
print(ratings_movies.info())

all_genres = movies["genres"].str.split("|").explode()

# Count unique genre
unique_genres = all_genres.unique()
print("Number of unique genres:", len(unique_genres))
print("Genres:\n", unique_genres)
ratings_movies.to_csv("Master_large.csv", index=False)
print("Done!")







