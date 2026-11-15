import pandas as pd
import numpy as np

def generate_dataset(num_samples=500):
    np.random.seed(42)
    # Simulate study hours between 1 and 10 hours
    hours_studied = np.random.uniform(1, 10, num_samples)
    
    # Simulate exam scores based on hours studied + some random noise
    exam_score = 40 + (hours_studied * 6.0) + np.random.normal(0, 3.5, num_samples)
    exam_score = np.clip(exam_score, 0, 100) # Keep scores between 0 and 100
    
    df = pd.DataFrame({
        'Hours_Studied': hours_studied,
        'Exam_Score': exam_score
    })
    
    df.to_csv('Student_Performance.csv', index=False)
    print("✅ 'Student_Performance.csv' generated successfully!")

if __name__ == '__main__':
    generate_dataset()