import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

def load_data(file_path):
    """Loads the movie dataset from a CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        sys.exit(1)
    return pd.read_csv(file_path)

def get_recommendations(user_preferences, df):
    """
    Matches user preferences with movies using cosine similarity.
    """
    # Combine user preferences with the movie genres for vectorization
    # We create a temporary dataframe to include the user's preference as a "movie"
    temp_df = pd.concat([df, pd.DataFrame({'title': ['User'], 'genres': [user_preferences]})], ignore_index=True)
    
    # Vectorize the genres
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(temp_df['genres'])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(count_matrix)
    
    # Get similarity scores for the user preference (last row)
    # Exclude the last row itself (the user preference)
    user_idx = len(temp_df) - 1
    similarity_scores = list(enumerate(similarity[user_idx]))
    
    # Sort movies based on similarity scores
    # Skip the user preference itself (last index)
    sorted_movies = sorted(similarity_scores[:-1], key=lambda x: x[1], reverse=True)
    
    # Return top 5 recommendations
    recommendations = []
    for i in range(5):
        movie_idx = sorted_movies[i][0]
        recommendations.append(df.iloc[movie_idx]['title'])
        
    return recommendations

def main():
    print("--- Movie Recommendation System ---")
    print("Loading database...")
    
    csv_path = 'movies.csv'
    df = load_data(csv_path)
    
    print("Database loaded successfully!")
    print("\nTell me what kind of movies you like (e.g., 'Action', 'Sci-Fi Comedy', 'Drama Romance').")
    print("Type 'exit' to quit.")

    try:
        while True:
            try:
                user_input = input("\nYour preferences: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break
                
            if user_input.lower() == 'exit':
                print("Goodbye! Enjoy your movies!")
                break
            
            if not user_input:
                print("Please enter some genres so I can recommend something!")
                continue
            
            print(f"Finding recommendations for: {user_input}...")
            
            try:
                recommendations = get_recommendations(user_input, df)
                print("\nTop 5 Recommendations for you:")
                for i, movie in enumerate(recommendations, 1):
                    print(f"{i}. {movie}")
            except Exception as e:
                print(f"An error occurred while matching patterns: {e}")
                
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
