import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

#Load ratings
ratings = pd.read_csv(
    "dataset/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
)

#Drop timestamp (when the rating was made)
ratings = ratings.drop(columns=["Timestamp"])

#Load movies
movies = pd.read_csv(
    "dataset/ml-1m/movies.dat", 
    sep="::", 
    engine='python', 
    encoding='latin1', 
    names=["MovieID", "Title", "Genres"]
)


#Merge datasets
df = pd.merge(ratings, movies, on="MovieID")
print (df.head())

#Create a pivot table
ratings_matrix = df.pivot_table(
    index='MovieID',   
    columns='UserID',  
    values='Rating'   
)
ratings_matrix_filled = ratings_matrix.dropna(how='all')

#Handle NaNs
ratings_matrix_filled = ratings_matrix.fillna(0)
print (ratings_matrix_filled.head())

# Apply PCA to reduce dimensionality for clustering
pca = PCA(n_components=0.9)  # You can tweak this
ratings_pca = pca.fit_transform(ratings_matrix_filled)

# Optional: Visualize explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.grid(True)
plt.show()

print(ratings_pca.shape) # 935 dimensions (down from 6000+)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(ratings_pca)
clusters = kmeans.labels_

# Merge the movies dataset with the clusters based on MovieID
movies_clustered = movies[movies['MovieID'].isin(ratings_matrix_filled.index)].copy()
movies_clustered['Cluster'] = clusters

# Now you can print or use the clustered movies DataFrame
print(movies_clustered.head())

score = silhouette_score(ratings_pca, clusters)
print(f'Silhouette Score: {score:.2f}')

# Join movie titles with clusters
clustered_movies = pd.merge(movies, ratings.groupby("MovieID").agg({'Rating': 'mean'}), on="MovieID")


# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
ratings_2d = pca_2d.fit_transform(ratings_matrix_filled)

# Plot the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=ratings_2d[:,0], y=ratings_2d[:,1], hue=clusters, palette="tab10", legend="full")
plt.title("Movie Clusters (2D PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

#sillohuette testing
silhouette_scores = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(ratings_pca)
    score = silhouette_score(ratings_pca, kmeans.labels_)
    silhouette_scores.append(score)

plt.plot(range(2, 15), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()