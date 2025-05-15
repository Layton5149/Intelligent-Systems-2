import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify, render_template

#flask setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Load ratings
ratings = pd.read_csv(
    "dataset/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
)
print(ratings.head())
ratings = ratings.drop(columns=["Timestamp"])

# Load movies
movies = pd.read_csv(
    "dataset/ml-1m/movies.dat",
    sep="::",
    engine="python",
    encoding="latin1",
    names=["MovieID", "Title", "Genres"]
)
print(movies.head())

# Merge ratings with movie info
Combined_df = pd.merge(ratings, movies, on="MovieID")
print (Combined_df.head())

# Create movie-user matrix (rows = movies, columns = users)
ratings_matrix = Combined_df.pivot_table(
    index='MovieID',
    columns='UserID',
    values='Rating'
)
print (ratings_matrix.head())

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

input_title = "Star Wars: Episode IV - A New Hope (1977)"  # Example input title

# Find MovieID from title
@app.route('/search', methods=['POST'])
def recommend():
    print ("Request received")
    data = request.get_json()
    input_title = data.get('title') 

    endList = []
    input_movie = movies[movies['Title'] == input_title]
    if input_movie.empty:
        return jsonify({'error': f"Movie '{input_title}' not found."})
    else:
        input_id = input_movie['MovieID'].values[0]

        # Find row index in the ratings matrix
        if input_id not in ratings_matrix_filled.index:
            return jsonify({'error': f"No ratings found for '{input_title}'."})
        else:
            movie_idx = ratings_matrix_filled.index.get_loc(input_id)
            _, indices = model_knn.kneighbors(
                ratings_pca[movie_idx].reshape(1, -1), n_neighbors=6
            )

            for i in range(1, 6):  # skip the first (itself)
                similar_idx = ratings_matrix_filled.index[indices[0][i]]
                endList.append(movie_id_to_title[similar_idx])

            print (endList)
            
            return jsonify({'input': input_title, 'recommendations': endList})

app.run(debug=True)