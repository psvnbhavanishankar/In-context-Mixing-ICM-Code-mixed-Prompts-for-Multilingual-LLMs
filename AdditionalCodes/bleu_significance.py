import pandas as pd
# from evaluate import load
from scipy import stats
from nltk.translate.bleu_score import sentence_bleu
import string

# Load data from the CSV files
df1 = pd.read_csv('MT0_xxl_ape/result_mr')
df2 = pd.read_csv('MT0_xxl_ape/result_mr_50p')
df_reference = pd.read_csv('MT0_xxl_ape/result_mr')

# bleu = load("sacrebleu")

sentences1 = df1['pred_label']
sentences2 = df2['pred_label']
reference_sentences = df_reference['ref']

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

# Calculate BLEU scores
def calculate_bleu(sentences, reference):
    return [sentence_bleu([reference[i]], sentences[i]) for i in range(len(sentences))]

sentences1 = [process_sentence(s) for s in list(sentences1)]
sentences2 = [process_sentence(s) for s in list(sentences2)]
reference_sentences = [process_sentence(s) for s in list(reference_sentences)]

bleu_scores1 = calculate_bleu(sentences1, reference_sentences)
bleu_scores2 = calculate_bleu(sentences2, reference_sentences)

# Check for normality
def check_normality(data):
    stat, p = stats.shapiro(data)
    if p > 0.05:
        return True
    else:
        return False

is_normal1 = check_normality(bleu_scores1)
is_normal2 = check_normality(bleu_scores2)

# Check for equal variances
def check_variance(data1, data2):
    stat, p = stats.levene(data1, data2)
    if p > 0.05:
        return True
    else:
        return False

is_equal_var = check_variance(bleu_scores1, bleu_scores2)

# Decide and perform the significance test
def perform_significance_test():
    if is_normal1 and is_normal2:
        if is_equal_var:
            t_stat, p = stats.ttest_ind(bleu_scores1, bleu_scores2)
            return "T-test", p
        else:
            t_stat, p = stats.ttest_ind(bleu_scores1, bleu_scores2, equal_var=False)
            return "Welch's T-test", p
    else:
        u_stat, p = stats.mannwhitneyu(bleu_scores1, bleu_scores2)
        return "Mann-Whitney U test", p

test_name, p_value = perform_significance_test()

# Output results
print(f"Test used: {test_name}")
print(f"P-value: {p_value}")
if p_value < 0.05:
    print("The difference in BLEU scores is statistically significant.")
else:
    print("The difference in BLEU scores is not statistically significant.")

