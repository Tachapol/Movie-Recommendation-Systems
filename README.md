# Movie Recommendation System

This project implements a movie recommendation system using various techniques, including content-based filtering and collaborative filtering (using Singular Value Decomposition - SVD).

## Project Overview
The goal of this project is to recommend movies to users based on their viewing history or the characteristics of movies they enjoy. It explores different approaches to build a robust recommendation engine.

## Data
The project utilizes two primary datasets:
- `ratings.csv`: Contains user ratings for various movies. Each row includes `userId`, `movieId`, `rating`, and `timestamp`.
- `movies.csv`: Contains movie metadata, including `movieId`, `title`, and `genres`.

## Exploratory Data Analysis (EDA)
Initial analysis was performed to understand the characteristics of the datasets:
- **Number of Ratings, Movies, and Users**: Provides a basic understanding of the dataset's scale.
- **Rating Distribution**: Visualizes the spread of ratings given by users.
- **Mean Global Rating**: Calculates the average rating across all movies.
- **Mean Rating Per User**: Determines the average rating given by each user.
- **Most/Least Frequently Rated Movies**: Identifies popular and unpopular movies based on the number of ratings.
- **Highest/Lowest Rated Movies**: Finds movies with the highest and lowest average ratings (with Bayesian averaging to account for movies with few ratings).
- **Genre Analysis**: Explores the distribution and frequency of movie genres.

## Recommendation Approaches

### 1. Item-Item Recommendations with k-Nearest Neighbors (kNN)
This approach uses the `ratings` data to create a user-item interaction matrix. For a given movie, it finds other movies that are "closest" in terms of user ratings using kNN with cosine or Euclidean similarity.

- `create_X(df)`: Generates a sparse matrix (user-item utility matrix) and mapper dictionaries for efficient lookups.
- `find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k, metric)`: Identifies `k` similar movies based on collaborative filtering.

### 2. Content-Based Recommendations (Genre-based Similarity)
This method recommends movies based on their genre similarity. It creates a binary representation of movie genres and then calculates cosine similarity between movies based on their genre vectors.

- **Genre One-Hot Encoding**: Converts the `genres` column into a binary representation where each genre is a separate column.
- `cosine_similarity(movie_genres, movie_genres)`: Computes the cosine similarity matrix between all movies based on their genres.
- `movie_finder(title)`: Uses fuzzy matching (`fuzzywuzzy`) to find the closest movie title in the dataset.
- `get_content_based_recommendations(title_string, n_recommendations)`: Provides `n` content-based recommendations for a given movie title.

### 3. Singular Value Decomposition (SVD)
SVD is a collaborative filtering technique that reduces the dimensionality of the user-item interaction matrix, capturing latent factors that represent user preferences and movie characteristics. These latent factors are then used to find similar movies.

- `TruncatedSVD`: Applied to the transposed user-item matrix to obtain a lower-dimensional representation of movies (`Q`).
- `recommendation(title_string)`: Utilizes the SVD-reduced matrix `Q` and kNN with cosine similarity to provide recommendations.

## How to Use

To get recommendations, you can use the following functions:

### Item-Item Recommendations (kNN on user-item matrix):
```python
movie_id = 1 # Example movieId for 'Toy Story (1995)'
similar_movies_knn = find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, metric='cosine', k=10)
print(f"Because you watched {movie_titles[movie_id]} we recommend :")
for mid in similar_movies_knn:
  print(movie_titles[mid])
```

### Content-Based Recommendations (Genre-based):
```python
get_content_based_recommendations('Jumanji', 10)
```

### SVD-Based Recommendations:
```python
recommendation('Jumanji')
```

Replace `'Jumanji'` or `movie_id` with your desired movie to get recommendations.
