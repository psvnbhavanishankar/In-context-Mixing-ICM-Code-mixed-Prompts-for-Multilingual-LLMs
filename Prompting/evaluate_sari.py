from evaluate import load
import pandas as pd
import string

# Load SARI metric
sari = load("sari")

# Read the CSV
df = pd.read_csv("MT0_xxl_results/result_pt_80p")

def process_sentence(sentence):
    if not isinstance(sentence, str):
        return ""
    sentence = sentence.split('\n')[0]
    sentence = sentence.strip().lower()
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, "")
    sentence = sentence.strip()
    if sentence and sentence[-1] == '।':
        sentence = sentence[:-1]
    return sentence

# Process predictions
original = [process_sentence(s) for s in df['original']]
predicted = [process_sentence(s) for s in df['pred_label']]

# Assuming columns "ref1", "ref2", ... "refN" are reference columns
# Change ["ref1", "ref2", "refN"] to your actual column names
reference_columns = ["label1", "label2", "label3", "label4"]
references = []

for _, row in df.iterrows():
    current_references = [process_sentence(row[col]) for col in reference_columns]
    references.append(current_references)

# Compute SARI score
results = {}
results['sari'] = sari.compute(sources=original, predictions=predicted, references=references)
print(results)
