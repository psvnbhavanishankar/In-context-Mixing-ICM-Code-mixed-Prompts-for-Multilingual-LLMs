import requests
import os

def download_dictionary():
    # Download the German-English dictionary from the provided link
    url = "https://dl.fbaipublicfiles.com/arrival/dictionaries/pt-en.txt"
    response = requests.get(url, stream=True)
    filename = "de-en.txt"
    
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    # Save the dictionary with a more descriptive name
    os.rename(filename, "portuguese_english_dict.txt")

if __name__ == "__main__":
    download_dictionary()
    print("German-English dictionary saved as 'german_english_dict.txt'")
