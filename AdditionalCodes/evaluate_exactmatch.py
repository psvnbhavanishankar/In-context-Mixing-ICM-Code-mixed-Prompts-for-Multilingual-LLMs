from evaluate import load
import pandas as pd
import string

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

exact_match_metric = load("exact_match")
bleu = load("sacrebleu")
# meteor = load('meteor')
# comet = load('comet')
# bertscore = load('bertscore')

# import torch

# # Check if CUDA (GPU) is available
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print("Using GPU:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device('cpu')
#     print("Using CPU")


# # Optimize for Tensor Cores if available
# if 'A100' in torch.cuda.get_device_name(0):
#     # Set the precision for matrix multiplications
#     # Choose 'medium' for a balance between performance and precision
#     # Or 'high' if you need higher precision
#     torch.set_float32_matmul_precision('medium')



df = pd.read_csv("MT0_xxl_results/result_m_eng_l")
reference = list(df.label)
predicted = list(df.pred_label)
# source = list(df.disfluent)

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

reference = [process_sentence(s) for s in list(df.label)]
# source = [process_sentence(s) for s in list(df.disfluent)]
predicted = [process_sentence(s) for s in list(df.pred_label)]





results = {}
results['exact_match'] = exact_match_metric.compute(predictions=predicted, references=reference)
results['bleu'] = bleu.compute(predictions=predicted, references=reference)
# results['meteor'] = meteor.compute(predictions=predicted, references=reference)
# results['comet'] = comet.compute(sources=source, predictions=predicted, references=reference)
# results['bertscore'] = bertscore.compute(predictions=predicted, references=reference)

print(results)