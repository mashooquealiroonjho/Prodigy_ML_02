import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load your customer data
data = pd.read_csv('Mall_Customers.csv')  # Replace with your dataset file path

# Select relevant features
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (K)
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# Based on the Elbow Method, choose the number of clusters (K)
k = 5  # Replace with your chosen number of clusters

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Reduce dimensionality for visualization (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='rainbow')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Clusters')

# Visualize individual features (columns)
plt.figure(figsize=(15, 5))
for i, feature in enumerate(X.columns):
    plt.subplot(1, 3, i+1)
    plt.scatter(X[feature], X_pca[:, 1], c=data['Cluster'], cmap='rainbow')
    plt.xlabel(feature)
    plt.ylabel('PCA Component 2')
    plt.title(f'Customer Clusters by {feature}')

plt.tight_layout()
plt.show()

# Display cluster centers and statistics for each cluster
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_info = pd.DataFrame(cluster_centers, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
print(cluster_info)
