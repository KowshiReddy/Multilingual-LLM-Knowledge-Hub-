import easyocr
from PIL import Image
import re

# Initialize the OCR Reader
reader = easyocr.Reader(['en'])

# Function to extract text from image using OCR
def extract_text_from_image(image_path):
    try:
        result = reader.readtext(image_path)
        print(result)
        text = " ".join([res[1] for res in result])
        return text
    except Exception as e:
        print(f"Error reading the image: {e}")
        return ""

def parse_extracted_text(text):
    title_match = re.search(r"Title:\s*(.+?)\n", text)
    author_match = re.search(r"Author:\s*(.+?)\n", text)
    genre_match = re.search(r"Genre:\s*(.+?)\n", text)
    description_match = re.search(r"Description:\s*(.+?)\n", text)
    
    title = title_match.group(1) if title_match else "Unknown"
    author = author_match.group(1) if author_match else "Unknown"
    genre = genre_match.group(1) if genre_match else "Unknown"
    description = description_match.group(1) if description_match else "No description available"
    print(title,author,genre,description)
    return title, author, genre, description
