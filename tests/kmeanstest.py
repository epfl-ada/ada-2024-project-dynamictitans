from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def kmeans_elbow_test(data):
    """
    Finding sum-of-squared-errors elbow to determine the best k number for clustering
    :return: k-means elbow plot of sum-of-squares-error vs k number
    """
    x=data['averageRating']
    x = (x -x.mean())/x.std()
    y=data['Log_Revenue']
    y = (y -y.mean())/y.std()
    cleaned_data = data.copy()
    cleaned_data['averageRating'] = (x - np.min(x)) / (np.max(x) - np.min(x))
    cleaned_data['Log_Revenue'] = (y - np.min(y)) / (np.max(y) - np.min(y))
    new_data = np.column_stack((x, y))
    # #elbow
    sse = []
    k_range = range(1, 11)  
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(new_data)
        sse.append(kmeans.inertia_)  
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE (Inertia)')
    plt.show()


def kmeans_test(data, n_clusters):
    """
    Finding sum-of-squared-errors elbow to determine the best k number for clustering
    :return: k-means clustering map
    """
    x=data['averageRating']
    x = (x -x.mean())/x.std()
    y=data['Log_Revenue']
    y = (y -y.mean())/y.std()
    cleaned_data = data.copy()
    cleaned_data['averageRating'] = (x - np.min(x)) / (np.max(x) - np.min(x))
    cleaned_data['Log_Revenue'] = (y - np.min(y)) / (np.max(y) - np.min(y))
    new_data = np.column_stack((x, y))
    kmeans = KMeans(n_clusters, random_state=42)  
    cleaned_data['cluster'] = kmeans.fit_predict(new_data)  
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='averageRating', y='Log_Revenue', hue='cluster', palette='Set1', data=cleaned_data, s=100, edgecolor='black', marker='o')
    plt.title('K-means Clustering of Log Revenue vs IMDb Rating')
    plt.xlabel('Rating')
    plt.ylabel('Log Revenue')
    plt.legend(title='Cluster')
    plt.show()