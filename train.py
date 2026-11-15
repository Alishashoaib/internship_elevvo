import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 1. Load Data
    try:
        df = pd.read_csv('Student_Performance.csv')
    except FileNotFoundError:
        print("❌ Error: 'Student_Performance.csv' not found. Please run 'generate_dummy_data.py' first.")
        return

    print("📊 Dataset Preview:")
    print(df.head(), "\n")

    # 2. Define Features and Target
    X = df[['Hours_Studied']]
    y = df['Exam_Score']

    # 3. Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize and Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"🔧 Model Parameters:")
    print(f"   - Intercept (c): {model.intercept_:.2f}")
    print(f"   - Coefficient (m): {model.coef_[0]:.2f}\n")

    # 5. Make Predictions & Evaluate
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"📈 Evaluation Metrics:")
    print(f"   - R² Score: {r2:.4f}")
    print(f"   - Root Mean Squared Error (RMSE): {rmse:.4f}\n")

    # 6. Visualize and Save Plot
    plt.figure(figsize=(9, 5))
    plt.scatter(X_test, y_test, color='royalblue', label='Actual Test Scores', alpha=0.7)
    
    # Sort values for a clean regression line plot
    sort_idx = np.argsort(X_test['Hours_Studied'].values)
    plt.plot(X_test.iloc[sort_idx], y_pred[sort_idx], color='crimson', linewidth=2.5, label='Regression Line')
    
    plt.title('Student Score Prediction: Study Hours vs. Exam Score', fontsize=12, fontweight='bold')
    plt.xlabel('Hours Studied')
    plt.ylabel('Exam Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('prediction_plot.png', dpi=300, bbox_inches='tight')
    print("💾 Plot saved successfully as 'prediction_plot.png'.")
    plt.show()

if __name__ == '__main__':
    main()