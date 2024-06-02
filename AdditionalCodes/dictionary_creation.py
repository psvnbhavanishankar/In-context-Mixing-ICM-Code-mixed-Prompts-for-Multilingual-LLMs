import requests
import bz2
import xml.etree.ElementTree as ET
import os
import pickle
from tqdm import tqdm
import mwparserfromhell

# Step 1: Download the latest dump
DUMP_URL = "https://dumps.wikimedia.org/tewiktionary/latest/tewiktionary-latest-pages-articles.xml.bz2"
response = requests.get(DUMP_URL, stream=True)

print("Downloading the latest dump...")
total_size = int(response.headers.get('content-length', 0))
progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

dump_file = "tewiktionary-latest-pages-articles.xml.bz2"
with open(dump_file, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        progress_bar.update(len(chunk))
        file.write(chunk)
progress_bar.close()

# Step 2: Extract the dump
print("\nExtracting the dump...")
with bz2.open(dump_file, 'rb') as source, open(dump_file[:-4], 'wb') as dest:
    for line in source:
        dest.write(line)

# Step 3: Parse the XML dump and extract translations
print("Parsing the XML dump to extract translations...")
tree = ET.parse(dump_file[:-4])
root = tree.getroot()

ns = {'ns': 'http://www.mediawiki.org/xml/export-0.10/'}

translations = {}

for page in root.findall('ns:page', ns):
    title = page.find('ns:title', ns).text
    revision = page.find('ns:revision', ns)
    if revision:
        text_data = revision.find('ns:text', ns)
        if text_data and text_data.text:
            # Parse the wikitext
            wikicode = mwparserfromhell.parse(text_data.text)
            links = [link.title for link in wikicode.filter_wikilinks() if link.title.startswith("en:")]
            if links:
                english_translations = [str(link.split(':')[1]) for link in links]
                translations[title] = english_translations

# Display the first 1000 translations
print("\nDisplaying the first 1000 translations:")
for i, (telugu_word, english_words) in enumerate(translations.items()):
    if i >= 1000:
        break
    print(f"Telugu Word: {telugu_word}, English Translations: {', '.join(english_words)}")

# Save the translations to a pickle file
print("\nSaving translations to pickle file...")
pickle_filename = "telugu_english_translations.pkl"
with open(pickle_filename, 'wb') as file:
    pickle.dump(translations, file)

print(f"Translations saved to {pickle_filename}")

# Optional: Remove the downloaded files if you want
# os.remove(dump_file)
# os.remove(dump_file[:-4])
