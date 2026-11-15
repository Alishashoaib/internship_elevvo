import pandas as pd
import numpy as np

def generate_movie_dataset(num_users=50, num_movies=20, num_ratings=400):
    np.random.seed(42)
    
    user_ids = np.random.randint(1, num_users + 1, num_ratings)
    movie_ids = np.random.randint(1, num_movies + 1, num_ratings)
    
    # Generate realistic ratings between 1 and 5
    ratings = np.random.choice([1, 2, 3, 4, 5], size=num_ratings, p=[0.1, 0.1, 0.2, 0.35, 0.25])
    
    df = pd.DataFrame({
        'userId': user_ids,
        'movieId': movie_ids,
        'rating': ratings
    })
    
    # Drop duplicate user-movie pairs if any
    df = df.drop_duplicates(subset=['userId', 'movieId'])
    
    df.to_csv('movie_ratings.csv', index=False)
    print("✅ 'movie_ratings.csv' generated successfully!")

if __name__ == '__main__':
    generate_movie_dataset()