import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data from CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess the data
def preprocess_data(df):
    # Remove unwanted columns and any rows with missing data
    df.drop(columns=['isbn13', 'isbn10', 'subtitle'], errors='ignore', inplace=True)
    df.dropna(subset=['title', 'authors', 'categories', 'description'], inplace=True)
    return df

# Train the recommender system and save the model
def train_recommender(df):
    df['content'] = df['categories'] + " " + df['authors'] + " " + df['description']
    
    # TF-IDF Vectorizer to transform text data
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['content'])
    
    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Save the model (vectorizer and cosine similarity matrix) to a .pkl file in the root folder
    with open('book_recommendation_model.pkl', 'wb') as model_file:  # Save to the current directory
        pickle.dump((vectorizer, cosine_sim), model_file)

    print("Model saved as 'book_recommendation_model.pkl'.")


# Load pre-trained recommendation model
def load_model(model_filename='book_recommendation_model.pkl'):
    with open(model_filename, 'rb') as file:
        vectorizer, cosine_sim = pickle.load(file)
    return vectorizer, cosine_sim

# Recommend books based on cosine similarity (returns top N books)
def recommend_books(df, book_index=0, top_n=10):
    vectorizer, cosine_sim = load_model()
    if df.empty:
        return pd.DataFrame()

    # Get similarity scores for the selected book
    sim_scores = cosine_sim[book_index]
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    book_indices = [x[0] for x in sim_scores[1:top_n + 1]]
    
    # Return the top N recommended books as a DataFrame
    recommended_books = df.iloc[book_indices][['title', 'authors', 'categories', 'description']]
    
    return recommended_books

# df = load_data("books.csv")
# df = preprocess_data(df)
# train_recommender(df)