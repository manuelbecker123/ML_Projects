import kaggle
import os
from zipfile import ZipFile

def download_and_extract_data(save_path='data/raw'):
    """
    Downloads and extracts the AEP energy consumption dataset from Kaggle.
    """
    dataset_name = 'robikscube/hourly-energy-consumption'
    
    print(f"Downloading dataset: {dataset_name}")
    os.makedirs(save_path, exist_ok=True)
    
    # Download the dataset
    kaggle.api.dataset_download_files(dataset_name, path=save_path, quiet=False)
    
    # Unzip the downloaded file
    zip_path = os.path.join(save_path, f"{dataset_name.split('/')[1]}.zip")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)
        print(f"Extracted files to {save_path}")
        
    # Remove the zip file after extraction
    os.remove(zip_path)
    print("Removed zip file.")

if __name__ == '__main__':
    download_and_extract_data()