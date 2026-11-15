import pandas as pd
import numpy as np

def generate_sales_dataset(num_days=365):
    np.random.seed(42)
    
    # Generate date range for one year
    dates = pd.date_range(start='2025-01-01', periods=num_days, freq='D')
    
    # Base trend + weekly seasonality + random noise
    trend = np.linspace(50, 150, num_days)
    seasonality = 20 * np.sin(np.linspace(0, 3 * np.pi, num_days))
    noise = np.random.normal(0, 10, num_days)
    
    sales = trend + seasonality + noise
    sales = np.clip(sales, 10, None)
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales
    })
    
    df.to_csv('walmart_sales.csv', index=False)
    print("✅ 'walmart_sales.csv' generated successfully!")

if __name__ == '__main__':
    generate_sales_dataset()