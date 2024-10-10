from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import requests
from bs4 import BeautifulSoup

# Initialize Flask app
app = Flask(__name__)

# Define the label list for entity extraction (adjust based on your NER model)
label_list = ["O", "PRODUCT", "BRAND", "ORG", "MATERIAL", "COLOR"]

# Load the tokenizer and model for NER
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"  # A pre-trained model suitable for NER tasks
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

# Function to scrape text from a webpage
def scrape_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Function to extract products from the scraped text using NER
def extract_products_from_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
    
    extracted_products = [tokens[i] for i, label in enumerate(predictions) if label in {label_list.index('B-PRODUCT'), label_list.index('I-PRODUCT')}]
    return extracted_products

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    products = []
    url = ""
    if request.method == 'POST':
        url = request.form.get('url')
        text = scrape_text_from_url(url)
        products = extract_products_from_text(text)
    
    return render_template('index.html', products=products, url=url)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
