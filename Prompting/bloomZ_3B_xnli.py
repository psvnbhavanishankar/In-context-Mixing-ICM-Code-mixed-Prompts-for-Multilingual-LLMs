import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed, AutoModelForCausalLM
import os
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
import csv
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # SET the GPUs you want to use


torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = AutoModelForCausalLM.from_pretrained("models/bloomz-3b", use_cache=True)
tokenizer = AutoTokenizer.from_pretrained("models/bloomz-3b")

# set_seed(2023)

dataset = load_dataset('Divyanshu/indicxnli', 'te')
test_data = dataset['test'].shuffle()

resultList = []
predictions = []
ground_truth = []
count = 0

label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
output_map = {'entailment': 'entailment', 'contradiction': 'contradiction', 'neutral': 'neutral'}

for i in test_data:
    count += 1
    premise = i['premise']
    hypothesis = i['hypothesis']
    correct_label = label_map[i['label']]
    # print(premise, hypothesis)

    prompt_eng = (
        "Given the premise: 'One of our number will carry out your instructions minutely.' and the hypothesis: 'A member of my team will execute your orders with immense precision.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'Gays and lesbians.' and the hypothesis: 'Heterosexuals.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'He turned and smiled at Vrenna.' and the hypothesis: 'He smiled at Vrenna who was walking slowly behind him with her mother.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        "Given the premise: 'How do you know? All this is their information again.' and the hypothesis: 'This information belongs to them.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'States must show reasonable progress in their state implementation plans toward the congressionally mandated goal of returning to natural conditions in national parks and wilderness areas.' and the hypothesis: 'It is not necessary for there to be any improvement.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'She smiled back.' and the hypothesis: 'She was so happy she couldn't stop smiling.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: "
    )
    prompt_tel = (
        "Given the premise: 'మా నంబర్‌లో ఒకరు మీ సూచనలను సూక్ష్మంగా అమలు చేస్తారు.' and the hypothesis: 'నా బృందంలోని సభ్యుడు మీ ఆర్డర్‌లను చాలా ఖచ్చితత్వంతో అమలు చేస్తారు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'స్వలింగ సంపర్కులు మరియు లెస్బియన్లు.' and the hypothesis: 'భిన్న లింగ సంపర్కులు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'వేద వైపు తిరిగి నవ్వాడు.' and the hypothesis: 'తల్లితో కలిసి తన వెనకే మెల్లగా నడుస్తున్న వేదను చూసి నవ్వాడు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        "Given the premise: 'నీకు ఎలా తెలుసు ? ఇదంతా మళ్లీ వారి సమాచారం.' and the hypothesis: 'ఈ సమాచారం వారికే చెందుతుంది.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'జాతీయ ఉద్యానవనాలు మరియు నిర్జన ప్రాంతాలలో సహజ పరిస్థితులకు తిరిగి రావాలనే కాంగ్రెస్ నిర్దేశించిన లక్ష్యం వైపు రాష్ట్రాలు తమ రాష్ట్ర అమలు ప్రణాళికలలో సహేతుకమైన పురోగతిని చూపాలి.' and the hypothesis: 'ఏదైనా మెరుగుదల ఉండాల్సిన అవసరం లేదు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'ఆమె తిరిగి నవ్వింది.' and the hypothesis: 'ఆమె నవ్వు ఆపుకోలేక చాలా సంతోషించింది.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?"
    )
    prompt_t_mixed30 = (
        "Given the premise: 'మా నంబర్‌లో ఒకరు మీ సూచనలను Subtly Implemented చేస్తారు.' and the hypothesis: 'నా బృందంలోని సభ్యుడు మీ ఆర్డర్‌లను చాలా ఖచ్చితత్వంతో Implemented చేస్తారు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'Gay సంపర్కులు మరియు లెస్బియన్లు.' and the hypothesis: 'భిన్న లింగ సంపర్కులు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'వేద వైపు తిరిగి Laughing.' and the hypothesis: 'తల్లితో కలిసి తన వెనకే మెల్లగా walking says చూసి నవ్వాడు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        "Given the premise: 'నీకు ఎలా తెలుసు ? ఇదంతా మళ్లీ వారి సమాచారం.' and the hypothesis: 'ఈ సమాచారం వారికే చెందుతుంది.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'national ఉద్యానవనాలు మరియు నిర్జన ప్రాంతాలలో సహజ purpose Back రావాలనే కాంగ్రెస్ నిర్దేశించిన లక్ష్యం వైపు రాష్ట్రాలు తమ రాష్ట్ర అమలు ప్రణాళికలలో సహేతుకమైన పురోగతిని చూపాలి.' and the hypothesis: 'ఏదైనా మెరుగుదల ఉండాల్సిన అవసరం లేదు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'ఆమె తిరిగి నవ్వింది.' and the hypothesis: 'ఆమె నవ్వు ఆపుకోలేక చాలా happy.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?"
    )
    prompt_t_mixed50 = (
        "Given the premise: 'మా number ఒకరు మీ సూచనలను Subtly Implemented చేస్తారు.' and the hypothesis: 'నా group సభ్యుడు మీ ఆర్డర్‌లను very ఖచ్చితత్వంతో Implemented చేస్తారు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'Gays మరియు lesbians.' and the hypothesis: 'భిన్న sexuals?', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'వేద వైపు Back నవ్వాడు.' and the hypothesis: 'తల్లితో కలిసి తన Back మెల్లగా నడుస్తున్న Veda చూసి Laughing', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        "Given the premise: 'నీకు ఎలా తెలుసు ? ఇదంతా Again వారి information.' and the hypothesis: 'ఈ Information వారికే చెందుతుంది.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'National Parks మరియు నిర్జన regions సహజ purpose తిరిగి రావాలనే కాంగ్రెస్ directed objective side States తమ State Implemented ప్రణాళికలలో సహేతుకమైన పురోగతిని చూపాలి.' and the hypothesis: 'ఏదైనా మెరుగుదల ఉండాల్సిన contradiction need.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'ఆమె తిరిగి laughed.' and the hypothesis: 'ఆమె laughter ఆపుకోలేక చాలా happy.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?"
    )
    prompt_t_mixed80 = (
        "Given the premise: 'మా number ఒకరు మీ Instructions Subtly Implemented చేస్తారు.' and the hypothesis: 'నా group Member మీ Orders చాలా ఖచ్చితత్వంతో అమలు చేస్తారు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'Gays మరియు lesbians.' and the hypothesis: 'heterosexuals.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'వేద వైపు Back Laughing.' and the hypothesis: 'mother Together తన Back Slowly running వేదను Seeing Laughing.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        "Given the premise: 'నీకు ఎలా know ? ఇదంతా Again వారి information.' and the hypothesis: 'ఈ Information వారికే belong.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'national ఉద్యానవనాలు మరియు Desolation regions natural purpose తిరిగి రావాలనే కాంగ్రెస్ directed objective side States తమ State Implemented ప్రణాళికలలో సహేతుకమైన progress show.' and the hypothesis: 'ఏదైనా Improvement ఉండాల్సిన contradiction need', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'ఆమె back laughed.' and the hypothesis: 'ఆమె laughter ఆపుకోలేక very happy.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?"
    )

    prompt_t_mixed30 = (
        "Given the premise: 'మా నంబర్‌లో ఒకరు మీ సూచనలను Subtly Implemented చేస్తారు.' and the hypothesis: 'నా బృందంలోని సభ్యుడు మీ ఆర్డర్‌లను చాలా ఖచ్చితత్వంతో Implemented చేస్తారు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
        "Given the premise: 'Gay సంపర్కులు మరియు లెస్బియన్లు.' and the hypothesis: 'భిన్న లింగ సంపర్కులు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
        "Given the premise: 'వేద వైపు తిరిగి Laughing.' and the hypothesis: 'తల్లితో కలిసి తన వెనకే మెల్లగా walking says చూసి నవ్వాడు.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
        f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?"
    )

    

   

    

    # prompt_t_mixed30 = f"మా నంబర్‌లో ఒకరు మీ సూచనలను Subtly Implemented చేస్తారు. Question: Does this imply that నా బృందంలోని సభ్యుడు మీ ఆర్డర్‌లను చాలా ఖచ్చితత్వంతో Implemented చేస్తారు. ? Yes, no, or maybe?\n Output: yes\nGay సంపర్కులు మరియు లెస్బియన్లు. Question: Does this imply that భిన్న లింగ సంపర్కులు. ? Yes, no, or maybe?\n Output: no\nవేద వైపు తిరిగి Laughing. Question: Does this imply that తల్లితో కలిసి తన వెనకే మెల్లగా walking says చూసి నవ్వాడు. ? Yes, no, or maybe?\n Output: maybe\nనీకు ఎలా తెలుసు ? ఇదంతా మళ్లీ వారి సమాచారం. Question: Does this imply that ఈ సమాచారం వారికే చెందుతుంది. ? Yes, no, or maybe?\n Output: yes\nnational ఉద్యానవనాలు మరియు నిర్జన ప్రాంతాలలో సహజ purpose Back రావాలనే కాంగ్రెస్ నిర్దేశించిన లక్ష్యం వైపు రాష్ట్రాలు తమ రాష్ట్ర అమలు ప్రణాళికలలో సహేతుకమైన పురోగతిని చూపాలి. Question: Does this imply that ఏదైనా మెరుగుదల ఉండాల్సిన అవసరం లేదు. ? Yes, no, or maybe?\n Output: no\nఆమె తిరిగి నవ్వింది. Question: Does this imply that ఆమె నవ్వు ఆపుకోలేక చాలా happy. ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ? Yes, no, or maybe?\n Output:"

    prompt = prompt_eng
    input_ids = tokenizer(prompt, return_tensors="pt").to(0)
    sample = model.generate(**input_ids, max_length=1600,
                            top_k=0, temperature=0.0)

    # output = tokenizer.decode(sample[0])
    # print(output)

    if(tokenizer.decode(sample[0])[0] != '<pad>'):
        output = tokenizer.decode(sample[0]).split("</s>")[0] 
    else:
        output = tokenizer.decode(sample[0]).split("<pad>")[1].split("</s>")[0]
    
    print(output, correct_label)

    # prediction = output_map[output.strip().lower()]
    
    # predictions.append(prediction)
    ground_truth.append(correct_label)
    # resultList.append([premise, hypothesis, prediction, correct_label])

    if count % 10 == 0:
        print("{} instances done".format(count))


macro_precision, macro_recall, macro_fscore, _ = precision_recall_fscore_support(
    ground_truth, predictions, labels=['entailment', 'neutral', 'contradiction'], 
    average='macro', zero_division=0
)

# Compute accuracy
accuracy = sum([ground_truth[i] == predictions[i] for i in range(len(predictions))]) / len(predictions)

# Compute per-class metrics
precision, recall, fscore, support = precision_recall_fscore_support(
    ground_truth, predictions, labels=['entailment', 'neutral', 'contradiction'], 
    zero_division=0
)

# Create a pandas DataFrame for representation
df = pd.DataFrame({
    'Label': ['entailment', 'neutral', 'contradiction', 'macro-average'],
    'Precision': list(precision) + [macro_precision],
    'Recall': list(recall) + [macro_recall],
    'F-Score': list(fscore) + [macro_fscore],
    'Support': list(support) + ['N/A'],
    'Accuracy': list(['N/A'] * 3) + [accuracy]
})

print(df)


filename = 'BloomZ_3B_results/result_xnli_t_m30.csv'
fields = ['premise', 'hypothesis', 'prediction', 'correct_label']

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(resultList)
