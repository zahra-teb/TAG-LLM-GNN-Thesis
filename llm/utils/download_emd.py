
import requests
import os

# Define files and their target folders as a dictionary
files = {
    "crime_bert_base_uncased_512_cls_node.pt": {
        "url": "https://huggingface.co/datasets/ZhuofengLi/TEG-Datasets/resolve/main/goodreads_crime/emb/crime_bert_base_uncased_512_cls_node.pt",
        "folder": "goodreads_data/goodreads_crime/emb"
    }
}

def download_file(filename, url, folder, retries=3):
    """
    Downloads a file from the given URL and saves it to the specified folder.
    
    Args:
        filename (str): The name of the file to save as.
        url (str): The URL to download the file from.
        folder (str): The folder to save the file in.
        retries (int): The number of retries for failed downloads.
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Define the full path to save the file
    file_path = os.path.join(folder, filename)

    for attempt in range(retries):
        try:
            # Download the file
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"{filename} downloaded and saved at: {file_path}")
            return  # Exit the function if download succeeds

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                print(f"Failed to download {filename} after {retries} attempts.")
            else:
                print("Retrying...")

# Iterate through the files and download each one
for filename, info in files.items():
    download_file(filename, info["url"], info["folder"])
