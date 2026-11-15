import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Load Data
    try:
        df = pd.read_csv('Mall_Customers.csv')
    except FileNotFoundError:
        print("❌ Error: 'Mall_Customers.csv' not found. Please run 'generate_mall_data.py' first.")
        return

    print("📊 Dataset Preview:")
    print(df.head(), "\n")

    # 2. Select Features for Clustering (Income and Spending Score)
    X = df[['Annual_Income', 'Spending_Score']]

    # 3. Feature Scaling (Crucial for K-Means so features have equal weight)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Apply K-Means Clustering (Choosing 5 optimal clusters based on standard mall analysis)
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    print("📈 Cluster Centers (Scaled back or raw evaluation):")
    print(kmeans.cluster_centers_, "\n")

    # 5. Visualize the Clusters
    plt.figure(figsize=(9, 6))
    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    
    for i in range(5):
        cluster_data = df[df['Cluster'] == i]
        plt.scatter(
            cluster_data['Annual_Income'], 
            cluster_data['Spending_Score'], 
            s=50, c=colors[i], label=f'Cluster {i+1}', alpha=0.7
        )

    # Plot the cluster centroids (transformed back to original scale approximately for visualization)
    # For a simple display, we can plot the raw data points grouped by cluster color
    plt.title('Customer Segments: Annual Income vs. Spending Score', fontsize=12, fontweight='bold')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig('customer_clusters.png', dpi=300, bbox_inches='tight')
    print("💾 Cluster plot saved successfully as 'customer_clusters.png'.")
    plt.show()

if __name__ == '__main__':
    main()