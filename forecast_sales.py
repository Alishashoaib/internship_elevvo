import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 1. Load Data
    try:
        df = pd.read_csv('walmart_sales.csv')
    except FileNotFoundError:
        print("❌ Error: 'walmart_sales.csv' not found. Please run 'generate_sales_data.py' first.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Create Time-Based Features & Lag Features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Lag feature: previous day's sales
    df['Lag_1'] = df['Sales'].shift(1)
    
    # Drop rows with NaN values created by shifting
    df = df.dropna().reset_index(drop=True)

    print("📊 Dataset Preview with Time Features:")
    print(df.head(), "\n")

    # 3. Define Features (X) and Target (y)
    X = df[['DayOfWeek', 'Month', 'Day', 'Lag_1']]
    y = df['Sales']

    # 4. Time-aware Split (keeping chronological order: 80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = df['Date'].iloc[split_idx:]

    # 5. Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Make Predictions & Evaluate
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"📈 Evaluation Metrics:")
    print(f"   - R² Score: {r2:.4f}")
    print(f"   - RMSE: {rmse:.4f}\n")

    # 7. Visualize Actual vs. Predicted Sales Over Time
    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, y_test, label='Actual Sales', color='royalblue', alpha=0.8)
    plt.plot(test_dates, y_pred, label='Predicted Sales', color='crimson', linestyle='--', linewidth=2)
    
    plt.title('Sales Forecasting: Actual vs. Predicted Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('sales_forecast_plot.png', dpi=300, bbox_inches='tight')
    print("💾 Forecast plot saved successfully as 'sales_forecast_plot.png'.")
    plt.show()

if __name__ == '__main__':
    main()