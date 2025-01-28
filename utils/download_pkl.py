import requests
import os

files = {
    "crime.pkl": "https://huggingface.co/datasets/ZhuofengLi/TEG-Datasets/resolve/main/goodreads_crime/processed/crime.pkl"
}

download_folder_name = [
    "goodreads_data/goodreads_crime/processed"
]

def download_file(filename, url, folder):
    """
    Downloads a file from the given URL and saves it to the specified folder.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Define the full path to save the file
    file_path = os.path.join(folder, filename)
    
    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"{filename} downloaded and saved at: {file_path}")
    else:
        print(f"Failed to download {filename}. HTTP Status Code: {response.status_code}")

# Iterate through the files and download each one
for folder, url in zip(download_folder_name, files.values()):
    folder_name = folder
    download_file(os.path.basename(url), url, folder_name)
