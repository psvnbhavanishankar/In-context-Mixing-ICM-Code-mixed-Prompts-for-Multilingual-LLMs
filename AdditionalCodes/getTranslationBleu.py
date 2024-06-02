
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

from evaluate import load
import pandas as pd
import string
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # SET the GPUs you want to use

exact_match_metric = load("exact_match")
bleu = load("sacrebleu")

df = pd.read_csv("Annotations/ep1_transcripts.csv")
df2 = pd.read_csv("Annotations/ep1_translations.csv")
input_ = []
reference = []

# Step 3: Iterate through rows of the DataFrame and filter out rows with "contentType" as "overlap"
for index, row in df.iterrows():
    if row['contentType'] != 'overlap':
        # Append the values to input_ and reference if "contentType" is not "overlap"
        input_.append(row['asr_transcript'])
        reference.append(row['translation'])

# Load the NLLB model for translation
#tel_Telu
#hin_Deva
#mar_Deva
#ben_Beng
#vie_Latn
#ces_Latn
#por_Latn
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="tel_Telu", use_safetensors=True)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

def get_translation(word):
    """Fetch the English translation for a given Telugu word using the NLLB model."""
    inputs = tokenizer(word, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=1500
    )
    english_phrase = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    return english_phrase

# df = pd.read_csv("Annotations/ep1.csv")
# input_ = list(df.transcript)
# reference = list(df.translation)
# source = list(df.disfluent)

# # Step 1: Identify indices of "NULL" in input_
# null_indices = [i for i, transcript in enumerate(input_) if transcript.strip().lower() == "null"]

# # Step 2: Remove corresponding elements from input_ and reference
# input_ = [transcript for i, transcript in enumerate(input_) if i not in null_indices]
# reference = [translation for i, translation in enumerate(reference) if i not in null_indices]

def process_sentence(sentence):
    if not isinstance(sentence, str):
        return ""
    # Remove spaces before and after the sentence
    sentence = sentence.split('\n')[0]
    sentence = sentence.strip()
    sentence = sentence.lower()
    

    # Remove punctuation marks in the sentence
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, "")
    sentence = sentence.strip()
        
    if sentence == "":
        return sentence
    
    if (sentence[-1] == '।'):
        print(sentence)
        sentence = sentence[:-1]
        print(sentence)

    return sentence


processed_input = [process_sentence(s) for s in input_]
processed_ref = [process_sentence(s) for s in reference]
translated = []

for i in processed_input:
    translated_sentence = get_translation(i)
    print(translated_sentence)
    translated.append(process_sentence(translated_sentence))

results = {}
# results['exact_match'] = exact_match_metric.compute(predictions=predicted, references=reference)
results['bleu'] = bleu.compute(predictions=translated, references=processed_ref)
# results['meteor'] = meteor.compute(predictions=predicted, references=reference)
# results['comet'] = comet.compute(sources=source, predictions=predicted, references=reference)
# results['bertscore'] = bertscore.compute(predictions=predicted, references=reference)

print(results)
df2['cascaded_pred'] = translated



