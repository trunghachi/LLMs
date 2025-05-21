import os
import requests
import zipfile
import json

# Định nghĩa các thư mục lưu trữ dữ liệu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PRETRAINING_DIR = os.path.join(DATA_DIR, "pretraining")
FINETUNING_DIR = os.path.join(DATA_DIR, "finetuning")

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(PRETRAINING_DIR, exist_ok=True)
os.makedirs(FINETUNING_DIR, exist_ok=True)

def download_file(url, destination):
    """Tải file từ URL và lưu vào destination."""
    if not os.path.exists(destination):
        print(f"Downloading {url}...")
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {destination}")
    else:
        print(f"{destination} already exists, skipping download.")

def unzip_file(zip_path, extract_to):
    """Giải nén file zip vào thư mục chỉ định."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def download_datasets():
    """Tải các dataset cần thiết cho repository."""
    # Dataset 1: Wikitext-2 (dùng cho pretraining)
    wikitext_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    wikitext_zip_path = os.path.join(PRETRAINING_DIR, "wikitext-2-v1.zip")
    download_file(wikitext_url, wikitext_zip_path)
    unzip_file(wikitext_zip_path, PRETRAINING_DIR)
    
    # Dataset 2: Alpaca (dùng cho finetuning)
    alpaca_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    alpaca_path = os.path.join(FINETUNING_DIR, "alpaca_data.json")
    download_file(alpaca_url, alpaca_path)

if __name__ == "__main__":
    download_datasets()
