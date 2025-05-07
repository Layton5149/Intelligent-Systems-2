import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Load ratings
ratings = pd.read_csv(
    "dataset/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
)
ratings = ratings.drop(columns=["Timestamp"])

# Load movies
movies = pd.read_csv(
    "dataset/ml-1m/movies.dat",
    sep="::",
    engine="python",
    encoding="latin1",
    names=["MovieID", "Title", "Genres"]
)

# Merge ratings with movie info
df = pd.merge(ratings, movies, on="MovieID")

# Create movie-user matrix (rows = movies, columns = users)
ratings_matrix = df.pivot_table(
    index='MovieID',
    columns='UserID',
    values='Rating'
)

# Replace NaNs with the movie's average rating
ratings_matrix_filled = ratings_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

# Optional: Reduce dimensionality with PCA for speed
pca = PCA(n_components=50)
ratings_pca = pca.fit_transform(ratings_matrix_filled)

# Fit KNN model on PCA-reduced data
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(ratings_pca)

# Map movie IDs to titles for lookup
movie_id_to_title = movies.set_index('MovieID')['Title'].to_dict()

# --- Example: Recommend movies similar to a given title ---
input_title = "Mighty Morphin Power Rangers: The Movie (1995)"  # Change this to test other titles

# Find MovieID from title
input_movie = movies[movies['Title'] == input_title]
if input_movie.empty:
    print(f"Movie '{input_title}' not found.")
else:
    input_id = input_movie['MovieID'].values[0]

    # Find row index in the ratings matrix
    if input_id not in ratings_matrix_filled.index:
        print(f"No ratings found for '{input_title}'.")
    else:
        movie_idx = ratings_matrix_filled.index.get_loc(input_id)
        distances, indices = model_knn.kneighbors(
            ratings_pca[movie_idx].reshape(1, -1), n_neighbors=6
        )

        print(f"\nMovies similar to '{input_title}':")
        for i in range(1, len(indices[0])):  # skip the first (itself)
            similar_idx = ratings_matrix_filled.index[indices[0][i]]
            print(f"- {movie_id_to_title[similar_idx]}")
