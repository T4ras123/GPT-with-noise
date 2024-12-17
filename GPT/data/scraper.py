import os
import requests
import zipfile
import io
from PyPDF2 import PdfReader
from tqdm import tqdm  # Import tqdm

OUTPUT_DIR = "extracted_texts"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define repository details
REPO_URL = "https://github.com/hackerwhale/free-cybersecurity-ebooks"
ZIP_URL = "https://github.com/hackerwhale/free-cybersecurity-ebooks/archive/refs/heads/master.zip"
DOWNLOAD_DIR = "free-cybersecurity-ebooks"
PDF_DIR = os.path.join(DOWNLOAD_DIR, "pdf")

# Download and extract the repository
response = requests.get(ZIP_URL)
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(".")
os.rename("free-cybersecurity-ebooks-master", DOWNLOAD_DIR)

# Get list of PDF files
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

# Iterate over all PDFs and extract text with a progress bar
for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
    pdf_path = os.path.join(PDF_DIR, filename)
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_filename)
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
        
        print(f"\nExtracted text from {filename} to {txt_filename}")
    
    except Exception as e:
        print(f"\nFailed to extract {filename}: {e}")