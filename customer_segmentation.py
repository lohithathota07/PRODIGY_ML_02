import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig("elbow_method.png")
plt.show()

# Distribution Graphs

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

axes[0].hist(data['Age'], bins=10)
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age')

axes[1].hist(data['Annual Income (k$)'], bins=10)
axes[1].set_title('Annual Income Distribution')
axes[1].set_xlabel('Annual Income (k$)')

axes[2].hist(data['Spending Score (1-100)'], bins=10)
axes[2].set_title('Spending Score Distribution')
axes[2].set_xlabel('Spending Score')

plt.tight_layout()
plt.savefig("distributions.png")
plt.show()

# Silhouette Score Analysis

scores = []

for i in range(2, 11):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    score = silhouette_score(X, labels)
    scores.append(score)

plt.figure(figsize=(8,5))
plt.plot(range(2,11), scores, marker='o')
plt.title('Silhouette Score Analysis')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.savefig("silhouette_score.png")
plt.show()

best_clusters = scores.index(max(scores)) + 2
print("Best number of clusters:", best_clusters)

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Cluster Visualization
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)

plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.savefig("customer_segmentation.png")
plt.show()

print("Customer Segmentation Completed Successfully!")