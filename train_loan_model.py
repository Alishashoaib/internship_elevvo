import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    # 1. Load Data
    try:
        df = pd.read_csv('loan_data.csv')
    except FileNotFoundError:
        print("❌ Error: 'loan_data.csv' not found. Please run 'generate_loan_data.py' first.")
        return

    print("📊 Dataset Preview:")
    print(df.head(), "\n")

    # 2. Handle Missing Values
    # Fill numerical missing values with the median
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())
    
    # Fill categorical missing values with the mode (most frequent value)
    df['Employment_Status'] = df['Employment_Status'].fillna(df['Employment_Status'].mode()[0])

    # 3. Encode Categorical Features using One-Hot Encoding
    df = pd.get_dummies(df, columns=['Employment_Status', 'Education'], drop_first=True)

    # 4. Define Features (X) and Target (y)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train a Classification Model (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 7. Evaluate Performance
    y_pred = model.predict(X_test)

    print("📈 Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred), "\n")
    print("Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()