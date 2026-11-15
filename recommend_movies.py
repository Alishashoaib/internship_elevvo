import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # 1. Load Data
    try:
        df = pd.read_csv('movie_ratings.csv')
    except FileNotFoundError:
        print("❌ Error: 'movie_ratings.csv' not found. Please run 'generate_movie_data.py' first.")
        return

    print("📊 Dataset Preview:")
    print(df.head(), "\n")

    # 2. Create User-Item Matrix (Rows = Users, Columns = Movies, Values = Ratings)
    user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    print(f"User-Item Matrix Shape: {user_item_matrix.shape}\n")

    # 3. Compute User Similarity using Cosine Similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, 
        index=user_item_matrix.index, 
        columns=user_item_matrix.index
    )

    # 4. Recommend Movies for a Sample User (e.g., User ID 1)
    target_user_id = 1
    if target_user_id not in user_item_matrix.index:
        print(f"User {target_user_id} not found in matrix.")
        return

    print(f"🔍 Finding recommendations for User {target_user_id}...")

    # Find top 3 most similar users (excluding the user themselves)
    similar_users = user_similarity_df[target_user_id].sort_values(ascending=False).iloc[1:4].index

    # Get movies rated by similar users that the target user hasn't rated yet
    target_user_rated = user_item_matrix.loc[target_user_id]
    unrated_movies = target_user_rated[target_user_rated == 0].index

    # Calculate weighted average ratings from similar users
    recommendation_scores = {}
    for movie_id in unrated_movies:
        # Sum of ratings given by similar users weighted by their similarity score
        scores = user_item_matrix.loc[similar_users, movie_id]
        sim_scores = user_similarity_df.loc[target_user_id, similar_users]
        
        # Filter out users who haven't rated this movie (rating == 0)
        valid_idx = scores > 0
        if valid_idx.sum() > 0:
            predicted_rating = np.dot(scores[valid_idx], sim_scores[valid_idx]) / sim_scores[valid_idx].sum()
            recommendation_scores[movie_id] = predicted_rating

    # Sort recommendations by highest predicted rating
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n🎬 Top Movie Recommendations for User {target_user_id}:")
    for movie_id, score in sorted_recommendations[:5]:
        print(f"   - Movie ID: {movie_id} (Estimated Score: {score:.2f})")

if __name__ == '__main__':
    main()