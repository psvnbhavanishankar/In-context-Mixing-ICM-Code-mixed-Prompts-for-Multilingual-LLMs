import csv
import string
from scipy.stats import wilcoxon
import numpy as np

def process_sentence(sentence):
    if not isinstance(sentence, str):
        return ""
    
    sentence = sentence.split('\n')[0]
    sentence = sentence.strip()
    sentence = sentence.lower()
    
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, "")
    sentence = sentence.strip()
    
    if sentence and sentence[-1] == '।':
        sentence = sentence[:-1]

    return sentence

# Read CSV and generate exact match scores for Prompt A
with open('MT0_xxl_results/result_vi', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    scores_a = [1 if process_sentence(row['pred_label']) == process_sentence(row['label']) else 0 for row in reader]

# Read CSV and generate exact match scores for Prompt B
with open('MT0_xxl_results/result_vi_80p', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    scores_b = [1 if process_sentence(row['pred_label']) == process_sentence(row['label']) else 0 for row in reader]

# Count the number of 1s in each list
count_a = scores_a.count(1)
count_b = scores_b.count(1)

# Print the counts
print(f"Number of exact matches for Prompt A: {count_a}")
print(f"Number of exact matches for Prompt B: {count_b}")

# Conduct Wilcoxon Signed Rank test
w_stat, p_val = wilcoxon(scores_a, scores_b)

# Print the results
print(f"Wilcoxon Signed Rank statistic: {w_stat}")
print(f"P-value: {p_val}")

if p_val < 0.05:
    print("The difference in score distributions between the prompts is statistically significant.")
else:
    print("The difference in score distributions between the prompts is not statistically significant.")
