import pandas as pd
import numpy as np

def generate_loan_dataset(num_samples=500):
    np.random.seed(42)
    
    income = np.random.normal(50000, 20000, num_samples)
    income = np.clip(income, 15000, 150000)
    
    loan_amount = np.random.normal(150000, 50000, num_samples)
    loan_amount = np.clip(loan_amount, 20000, 400000)
    
    credit_history = np.random.choice([0.0, 1.0], size=num_samples, p=[0.2, 0.8])
    employment = np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], size=num_samples, p=[0.6, 0.3, 0.1])
    education = np.random.choice(['Graduate', 'Not Graduate'], size=num_samples, p=[0.75, 0.25])
    
    # Introduce some missing values intentionally to practice data cleaning
    employment_missing = np.where(np.random.rand(num_samples) < 0.05, None, employment)
    credit_missing = np.where(np.random.rand(num_samples) < 0.05, np.nan, credit_history)

    # Simple logic for approval: High income, good credit history, and employment increase approval chances
    approval_score = (income / 50000) * 0.4 + (credit_missing * 0.5) - (loan_amount / 300000) * 0.2
    approval_score += np.random.normal(0, 0.2, num_samples)
    loan_status = np.where(approval_score > 0.5, 1, 0)

    df = pd.DataFrame({
        'ApplicantIncome': income,
        'LoanAmount': loan_amount,
        'Credit_History': credit_missing,
        'Employment_Status': employment_missing,
        'Education': education,
        'Loan_Status': loan_status
    })

    df.to_csv('loan_data.csv', index=False)
    print("✅ 'loan_data.csv' generated successfully with missing values!")

if __name__ == '__main__':
    generate_loan_dataset()