import random
import spacy

# Load the German spacy model
# nlp = spacy.load('de_core_news_sm')
# nlp = spacy.load('de_core_news_sm')
# nlp = spacy.load('tr_core_news_trf') #French
# nlp = spacy.load('hi_core_news_sm') #Greek
# nlp = spacy.load("fr_core_news_sm")
nlp = spacy.load('pt_core_news_sm')
# nlp = spacy.load('es_core_news_sm')

def load_german_english_dict(file_path):
    """
    Load the German-English dictionary from a file.
    
    Args:
    - file_path (str): Path to the dictionary file.
    
    Returns:
    - dict: German-English dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return {line.split()[0]: line.split()[1] for line in lines}

def translate_content_words(sentence, dictionary, probability=0.5):
    """
    Randomly translate content words from German to English.
    
    Args:
    - sentence (str): German sentence to translate.
    - dictionary (dict): Bilingual German-English dictionary.
    - probability (float): Probability to translate a word.
    
    Returns:
    - str: Sentence with randomly translated content words.
    """
    doc = nlp(sentence.lower())
    translated_sentence = []

    for token in doc:
        # Check if the token is a content word
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', ]:
            # Randomly decide whether to translate
            if random.random() < probability:
                # Translate if word is in the dictionary, otherwise keep the original word
                translated_sentence.append(dictionary.get(token.text, token.text))
            else:
                translated_sentence.append(token.text)
        else:
            translated_sentence.append(token.text)

    return ' '.join(translated_sentence)

# Load the dictionary from the file
german_english_dict = load_german_english_dict('Dictionary/portuguese_english_dict.txt')

# Example usage
sentence = "비교 가능한 유속을 유지할 수있을 때 그 결과가 높습니다."
print(translate_content_words(sentence, german_english_dict, 0.5))
print(translate_content_words(sentence, german_english_dict, 0.8))
print(translate_content_words(sentence, german_english_dict, 1.0))
