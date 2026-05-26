# Project 3: AI Movie Recommendation System

A simple content-based recommendation system that understands user preferences, matches patterns using cosine similarity, and recommends relevant movies.

## Features
- **User Preference Input:** Takes genres or keywords from the user.
- **Pattern Matching:** Uses `CountVectorizer` and `Cosine Similarity` to find the best matches in the database.
- **Top Recommendations:** Displays the top 5 most relevant movies.

## How it Works
1. The system loads a dataset of movies and their genres from `movies.csv`.
2. The user provides their preferred genres (e.g., "Action Sci-Fi").
3. The system converts the genres into vectors.
4. It calculates the similarity between the user's input vector and each movie's genre vector.
5. The movies with the highest similarity scores are returned.

## Requirements
- Python 3.x
- pandas
- scikit-learn

## Setup
1. Navigate to the project directory:
   ```bash
   cd DecodeLabs-Internship_project-3
   ```
2. Install dependencies (if not already installed):
   ```bash
   pip install pandas scikit-learn
   ```
3. Run the recommender:
   ```bash
   python recommender.py
   ```

## Example Usage
```text
--- Movie Recommendation System ---
Loading database...
Database loaded successfully!

Tell me what kind of movies you like (e.g., 'Action', 'Sci-Fi Comedy', 'Drama Romance').
Type 'exit' to quit.

Your preferences: Action Sci-Fi
Finding recommendations for: Action Sci-Fi...

Top 5 Recommendations for you:
1. The Matrix
2. Inception
3. Mad Max: Fury Road
4. Blade Runner 2049
5. Arrival
```
