import pandas as pd
import numpy as np

def generate_customer_dataset(num_samples=300):
    np.random.seed(42)
    # Simulate Annual Income (k$) and Spending Score (1-100)
    # Creating distinct clusters for realistic K-Means grouping
    cluster1_income = np.random.normal(30, 8, num_samples // 3)
    cluster1_spending = np.random.normal(80, 10, num_samples // 3)

    cluster2_income = np.random.normal(85, 10, num_samples // 3)
    cluster2_spending = np.random.normal(80, 10, num_samples // 3)

    cluster3_income = np.random.normal(55, 12, num_samples - len(cluster1_income) - len(cluster2_income))
    cluster3_spending = np.random.normal(50, 15, len(cluster3_income))

    income = np.concatenate([cluster1_income, cluster2_income, cluster3_income])
    spending = np.concatenate([cluster1_spending, cluster2_spending, cluster3_spending])

    # Clip values to realistic ranges
    income = np.clip(income, 10, 140)
    spending = np.clip(spending, 1, 100)

    df = pd.DataFrame({
        'CustomerID': range(1, num_samples + 1),
        'Annual_Income': income,
        'Spending_Score': spending
    })

    df.to_csv('Mall_Customers.csv', index=False)
    print("✅ 'Mall_Customers.csv' generated successfully!")

if __name__ == '__main__':
    generate_customer_dataset()