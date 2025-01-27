from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import pandas as pd
from utils.ocr import extract_text_from_image, parse_extracted_text
from data.book_recommender import recommend_books  

app = Flask(__name__)
CORS(app)

# Create upload directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
    
# Load pre-trained recommendation model
try:
    with open('book_recommendation_model.pkl', 'rb') as file:
        vectorizer, cosine_sim = pickle.load(file)
except FileNotFoundError:
    print("Model file 'book_recommendation_model.pkl' not found. Please ensure it exists.")
    exit(1)

# Route for uploading image
@app.route('/upload', methods=['GET'])
def upload_image():
    try:
        # Check if the request contains a file
        # if 'image' not in request.files:
        #     return jsonify({'error': 'No image file provided'}), 400
        
        #file = request.files['image']
        
        filename = "uploads/upload_image.jpg"
        # if os.path.exists(filename):
        #     print(f"File found: {filename}")
        # else:
        #     print("File not found:", filename)

        # file.save(filename)
        
        # OCR to extract text from the uploaded image
        text = extract_text_from_image(filename)
        title, author, genre, description = parse_extracted_text(text)

        # Ensure that the parsed data was correctly extracted
        if not all([title, author, genre, description]):
            return jsonify({'error': 'Failed to extract necessary book details from image'}), 400

        # Load dataset of books
        df = pd.read_csv('data/books.csv')
        
        # Check if necessary columns are in the DataFrame
        required_columns = ['title', 'authors', 'categories', 'description']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': f'Missing required columns in dataset. Ensure these columns: {required_columns} are present'}), 400
        
        # Call the recommend_books method to get the recommendations
        recommended_books = recommend_books(df, top_n=10)
        
        return jsonify({'recommended_books': recommended_books.to_dict(orient='records')})
        
    except Exception as e:
        # Error handling for unexpected issues
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
