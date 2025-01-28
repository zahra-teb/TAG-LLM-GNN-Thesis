import requests
import gzip
import os

# Files to download with their URLs
files = {
    "goodreads_book_genres_initial.json": "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_genres_initial.json.gz",
    "goodreads_books_mystery_thriller_crime.json.gz": "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_mystery_thriller_crime.json.gz",
    "goodreads_reviews_mystery_thriller_crime.json.gz": "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz"
}

# Specify a single folder for all downloads
download_folder = "data/goodreads_crime/raw"

# Function to download and extract files
def download_and_extract(filename, url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    response = requests.get(url)
    gz_path = os.path.join(folder, filename)
    
    # Save the downloaded .gz file
    with open(gz_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the .gz file to a .json file
    json_path = gz_path.replace('.gz', '')
    with gzip.open(gz_path, 'rb') as f_in:
        with open(json_path, 'wb') as f_out:
            f_out.write(f_in.read())
    
    print(f"{filename} downloaded and extracted at: {folder}")

# Iterate through files and download them to the same folder
for filename, url in files.items():
    download_and_extract(filename, url, download_folder)
