import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
import csv
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # SET the GPUs you want to use
from collections import defaultdict
import sys



torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = AutoModelForSeq2SeqLM.from_pretrained("models/mt0-xxl-mt", use_cache=True)
tokenizer = AutoTokenizer.from_pretrained("models/mt0-xxl-mt")

set_seed(2023)

dataset = load_dataset('xnli', 'fr')
# dataset = load_dataset('xnli', 'en')
test_data = dataset['test'].shuffle()

resultList = []
predictions = []
ground_truth = []
count = 0

label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
output_map = {'entailment': 'entailment', 'contradiction': 'contradiction', 'neutral': 'neutral', 'yes': 'entailment', 'no': 'contradiction', 'maybe': 'neutral'}
output_map_default = defaultdict(lambda: '', output_map)
for i in test_data:
    count += 1
    premise = i['premise']
    hypothesis = i['hypothesis']
    correct_label = label_map[i['label']]

    translate_prompt = f"Translate from French to English: L' écrémage conceptuel de la crème a deux dimensions fondamentales : le produit et la géographie.\nOutput: Conceptual cream skimming has two fundamental dimensions: product and geography.\nTranslate from French to English: Un de nos numéros vous fera suivre vos instructions minutieusement.\nOutput: One of our numbers will have you follow your instructions carefully.\nTranslate from French to English: {premise}\nOutput: "
    input_ids = tokenizer(translate_prompt, return_tensors="pt").to(0)
    sample = model.generate(**input_ids, max_length=1500)
    premise = tokenizer.decode(sample[0]).split("<pad>")[1].split("</s>")[0]

    translate_prompt = f"Translate from French to English: L' écrémage conceptuel de la crème a deux dimensions fondamentales : le produit et la géographie.\nOutput: Conceptual cream skimming has two fundamental dimensions: product and geography.\nTranslate from French to English: Un de nos numéros vous fera suivre vos instructions minutieusement.\nOutput: One of our numbers will have you follow your instructions carefully.\nTranslate from French to English: {hypothesis}\nOutput: "
    input_ids = tokenizer(translate_prompt, return_tensors="pt").to(0)
    sample = model.generate(**input_ids, max_length=1500)
    hypothesis = tokenizer.decode(sample[0]).split("<pad>")[1].split("</s>")[0]

    # print(premise)
    # print(hypothesis)
   

    # prompt_eng = (
    #     "Given the premise: 'One of our number will carry out your instructions minutely.' and the hypothesis: 'A member of my team will execute your orders with immense precision.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
    #     "Given the premise: 'Gays and lesbians.' and the hypothesis: 'Heterosexuals.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
    #     "Given the premise: 'He turned and smiled at Vrenna.' and the hypothesis: 'He smiled at Vrenna who was walking slowly behind him with her mother.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
    #     "Given the premise: 'How do you know? All this is their information again.' and the hypothesis: 'This information belongs to them.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: entailment\n"
    #     "Given the premise: 'States must show reasonable progress in their state implementation plans toward the congressionally mandated goal of returning to natural conditions in national parks and wilderness areas.' and the hypothesis: 'It is not necessary for there to be any improvement.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: contradiction\n"
    #     "Given the premise: 'She smiled back.' and the hypothesis: 'She was so happy she couldn't stop smiling.', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: neutral\n"
    #     f"Given the premise: '{premise}' and the hypothesis: '{hypothesis}', is the hypothesis an entailment, contradiction, or neutral with respect to the premise?\nOutput: "
    # )
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

    prompt_eng = f"One of our number will carry out your instructions minutely. Question: Does this imply that A member of my team will execute your orders with immense precision. ? Yes, no, or maybe?\n Output: yes\nGays and lesbians. Question: Does this imply that Heterosexuals. ? Yes, no, or maybe?\n Output: no\nHe turned and smiled at Vrenna. Question: Does this imply that He smiled at Vrenna who was walking slowly behind him with her mother. ? Yes, no, or maybe?\n Output: maybe\nHow do you know ? All this is their information again. Question: Does this imply that This information belongs to them. ? Yes, no, or maybe?\n Output: yes\nStates must show reasonable progress in their state implementation plans toward the congressionally mandated goal of returning to natural conditions in national parks and wilderness areas. Question: Does this imply that It is not necessary for there to be any improvement. ? Yes, no, or maybe?\n Output: no\nShe smiled back. Question: Does this imply that She was so happy she couldn't stop smiling. ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_hi = f"हमारा एक नंबर आपके निर्देशों का सूक्ष्मता से पालन करेगा। Question: Does this imply that मेरी टीम का एक सदस्य आपके आदेशों को अत्यंत सटीकता के साथ निष्पादित करेगा। ? Yes, no, or maybe?\n Output: yes\nसमलैंगिक और लेस्बियन। Question: Does this imply that विषमलैंगिक। ? Yes, no, or maybe?\n Output: no\nवह मुड़ा और वेदा की ओर देखकर मुस्कुराया। Question: Does this imply that वह वेदा को देखकर मुस्कुराया जो अपनी माँ के साथ उसके पीछे धीरे-धीरे चल रही थी। ? Yes, no, or maybe?\n Output: maybe\nआपको कैसे मालूम ? ये सब उनकी जानकारी है। Question: Does this imply that ये जानकारी उनकी है। ? Yes, no, or maybe?\n Output: yes\nराज्यों को राष्ट्रीय उद्यानों और जंगल क्षेत्रों में प्राकृतिक परिस्थितियों में लौटने के कांग्रेस द्वारा निर्धारित लक्ष्य की दिशा में अपनी राज्य कार्यान्वयन योजनाओं में उचित प्रगति दिखानी चाहिए। Question: Does this imply that इसमें कोई सुधार होना जरूरी नहीं है। ? Yes, no, or maybe?\n Output: no\nवह वापस मुस्कुराई। Question: Does this imply that वह इतनी खुश थी कि वह मुस्कुराना बंद नहीं कर पा रही थी। ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_hi_30p = f"हमारा एक number आपके निर्देशों का सूक्ष्मता से पालन करेगा। Question: Does this imply that मेरी टीम का एक सदस्य आपके orders को extreme सटीकता के साथ निष्पादित करेगा। ? Yes, no, or maybe?\n Output: yes\nसमलैंगिक और लेस्बियन। Question: Does this imply that विषमलैंगिक। ? Yes, no, or maybe?\n Output: no\nवह मुड़ा और वेदा की side देखकर मुस्कुराया। Question: Does this imply that वह वेदा को देखकर smiled जो अपनी mother के साथ उसके पीछे धीरे-धीरे चल रही थी। ? Yes, no, or maybe?\n Output: maybe\nआपको कैसे know ? ये सब उनकी जानकारी है। Question: Does this imply that ये जानकारी theirs है। ? Yes, no, or maybe?\n Output: yes\nstates को national उद्यानों और जंगल क्षेत्रों में natural परिस्थितियों में लौटने के कांग्रेस द्वारा निर्धारित लक्ष्य की direction में अपनी राज्य कार्यान्वयन योजनाओं में appropriate progress show चाहिए। Question: Does this imply that इसमें कोई improvement होना जरूरी नहीं है। ? Yes, no, or maybe?\n Output: no\nवह वापस smiled। Question: Does this imply that वह इतनी happy थी कि वह smiling बंद नहीं कर पा रही थी। ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_hi_50p = f"हमारा एक number आपके निर्देशों का सूक्ष्मता से follow करेगा। Question: Does this imply that मेरी टीम का एक member आपके orders को extreme accuracy के साथ execute करेगा। ? Yes, no, or maybe?\n Output: yes\ngay और लेस्बियन। Question: Does this imply that heterosexual ? Yes, no, or maybe?\n Output: no\nवह twist और वेदा की side देखकर मुस्कुराया। Question: Does this imply that वह वेदा को देखकर smiled जो अपनी mother के साथ उसके पीछे धीरे-धीरे go रही थी। ? Yes, no, or maybe?\n Output: maybe\nआपको कैसे know ? ये सब उनकी information है। Question: Does this imply that ये information उनकी है। ? Yes, no, or maybe?\n Output: yes\nstates को national उद्यानों और जंगल क्षेत्रों में natural परिस्थितियों में लौटने के congress द्वारा decided लक्ष्य की direction में अपनी राज्य कार्यान्वयन योजनाओं में appropriate progress show चाहिए। Question: Does this imply that इसमें कोई improvement होना need नहीं है। ? Yes, no, or maybe?\n Output: no\nवह वापस smiled। Question: Does this imply that वह इतनी happy थी कि वह smiling stop नहीं कर पा रही थी। ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_hi_80p = f"हमारा एक number आपके instructions का subtlety से follow करेगा। Question: Does this imply that मेरी team का one member आपके orders को extreme accuracy के साथ execute करेगा। ? Yes, no, or maybe?\n Output: yes\ngay और lesbian. Question: Does this imply that heterosexual ? Yes, no, or maybe?\n Output: no\nवह twist और वेदा की side seeing smiled. Question: Does this imply that वह वेदा को देखकर smiled जो अपनी mother के together उसके पीछे धीरे-धीरे go रही थी। ? Yes, no, or maybe?\n Output: maybe\nआपको कैसे know ? ये all उनकी information है। Question: Does this imply that ये information theirs है। ? Yes, no, or maybe?\n Output: yes\nstates को national parks और जंगल क्षेत्रों में natural conditions में लौटने के congress द्वारा decided goal की direction में अपनी राज्य कार्यान्वयन योजनाओं में appropriate progress show चाहिए। Question: Does this imply that इसमें कोई improvement होना need नहीं है। ? Yes, no, or maybe?\n Output: no\nवह return smiled। Question: Does this imply that वह इतनी happy थी कि वह smiling stop नहीं कर पा रही थी। ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_tr = f"Kavramsal krem kaymağını iki temel boyutu vardır - ürün ve coğrafya . Question: Does this imply that Ürün ve coğrafya , krem kaymağını işi yapan şey . ? Yes, no, or maybe?\n Output: maybe\nBizim biri senin bizat taşır . Question: Does this imply that Benim bir üyesi emirlerini muazzam bir hassasiyetle infaz edecek . ? Yes, no, or maybe?\n Output: yes\nGeyler ve lezbiyenler . Question: Does this imply that Heteroseksüeller? Yes, no, or maybe?\n Output: no\nBir set kabin kapısından girip yere düştüm. Question: Does this imply that Kapıdan girip düştüm . ? Yes, no, or maybe?\n Output: yes\n Yetişkinler ve çocuklar için eğlence .Question: Does this imply that Sadece çocuklar için eğlence . ? Yes, no, or maybe?\n Output: no\nDöndü ' ya döndü ve gülümsedi . Question: Does this imply that Annesinin yanında yavaş yavaş yürüyen remix ' a gülümsedi . ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_tr_30p = f"Kavramsal krem kaymağını iki basis boyutu vardır - ürün ve geography. Question: Does this imply that Ürün ve geography, krem kaymağını işi yapan şey . ? Yes, no, or maybe?\n Output: maybe\nBizim biri senin bizat carries. Question: Does this imply that Benim bir üyesi emirlerini enormous bir hassasiyetle infaz edecek . ? Yes, no, or maybe?\n Output: yes\nGeyler ve lesbians . Question: Does this imply that Heteroseksüeller? Yes, no, or maybe?\n Output: no\nBir set cabin kapısından girip yere düştüm. Question: Does this imply that Kapıdan girip düştüm . ? Yes, no, or maybe?\n Output: yes\n Adults ve çocuklar için eğlence .Question: Does this imply that Sadece çocuklar için eğlence . ? Yes, no, or maybe?\n Output: no\nDöndü ' ya returned ve gülümsedi . Question: Does this imply that Annesinin yanında slowly yürüyen remix ' a gülümsedi . ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_tr_50p = f"Kavramsal cream kaymağını iki basis boyutu vardır - ürün ve geography. Question: Does this imply that Ürün ve geography, cream kaymağını işi yapan şey . ? Yes, no, or maybe?\n Output: maybe\nBizim biri senin personally carries. Question: Does this imply that Benim bir üyesi emirlerini enormous bir hassasiyetle execution edecek . ? Yes, no, or maybe?\n Output: yes\nGays ve lezbiyenler . Question: Does this imply that Heterosexuals? Yes, no, or maybe?\n Output: no\nBir set cabin kapısından flu yere düştüm. Question: Does this imply that Kapıdan flu düştüm . ? Yes, no, or maybe?\n Output: yes\n Adults ve kids için eğlence .Question: Does this imply that Sadece kids için eğlence . ? Yes, no, or maybe?\n Output: no\nreturned ' ya returned ve gülümsedi . Question: Does this imply that Annesinin yanında slowly yürüyen remix ' a gülümsedi . ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_tr_80p = f"Kavramsal cream kaymağını iki basis boyutu vardır - product ve geography. Question: Does this imply that product ve geography, cream kaymağını işi yapan şey . ? Yes, no, or maybe?\n Output: maybe\nBizim biri senin personally carries. Question: Does this imply that Benim bir üyesi emirlerini enormous bir hassasiyetle execution will . ? Yes, no, or maybe?\n Output: yes\nGays ve lesbians . Question: Does this imply that Heterosexuals? Yes, no, or maybe?\n Output: no\nBir set cabin kapısından flu yere düştüm. Question: Does this imply that Kapıdan flu düştüm . ? Yes, no, or maybe?\n Output: yes\n Adults ve kids için entertainment .Question: Does this imply that Sadece kids için entertainment . ? Yes, no, or maybe?\n Output: no\nreturned ' ya returned ve smiled . Question: Does this imply that Annesinin yanında slowly yürüyen remix ' a gülümsedi . ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_fr = f"L' écrémage conceptuel de la crème a deux dimensions fondamentales : le produit et la géographie. Question: Does this imply that Le produit et la géographie sont ce qui fait travailler la crème de la crème? Yes, no, or maybe?\n Output: maybe\nUn de nos numéros vous fera suivre vos instructions minutieusement. Question: Does this imply that Un membre de mon équipe exécutera vos ordres avec une grande précision ? Yes, no, or maybe?\n Output: yes\nGays et lesbiennes . Question: Does this imply that Les hétérosexuels? Yes, no, or maybe?\n Output: no\nJ' ai traversé un ensemble de portes de cabine , et je suis tombé au sol. Question: Does this imply that J' ai traversé les portes et je suis tombé . ? Yes, no, or maybe?\n Output: yes\n Amusant pour les adultes et les enfants .Question: Does this imply that Amusant pour seulement les enfants . ? Yes, no, or maybe?\n Output: no\nIl s' est retourné et a souri à vrenna . Question: Does this imply that Il sourit à vrenna qui marchait lentement derrière lui avec sa mère . ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_fr_30p = f"L' écrémage conceptuel de la crème a deux dimensions fondamentales : le produit et la géographie. Question: Does this imply that Le produit et la géographie sont ce qui fait travailler la crème de la crème? Yes, no, or maybe?\n Output: maybe\nUn de nos numéros vous fera suivre vos instructions thoroughly. Question: Does this imply that Un membre de mon team exécutera vos ordres avec une grande précision ? Yes, no, or maybe?\n Output: yes\nGays et lesbiennes . Question: Does this imply that Les hétérosexuels? Yes, no, or maybe?\n Output: no\nJ' ai traversé un ensemble de portes de cabine , et je suis tombé au floor. Question: Does this imply that J' ai traversé les doors et je suis tombé . ? Yes, no, or maybe?\n Output: yes\n Amusant pour les adultes et les enfants .Question: Does this imply that Amusant pour seulement les enfants . ? Yes, no, or maybe?\n Output: no\nIl s' est returned et a souri à vrenna . Question: Does this imply that Il sourit à vrenna qui marchait slowly derrière lui avec sa mère . ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_fr_50p = f"L' écrémage conceptuel de la crème a deux dimensions fondamentales : le product et la geography. Question: Does this imply that Le product et la geography sont ce qui fait travailler la crème de la crème? Yes, no, or maybe?\n Output: maybe\nUn de nos numéros vous will suivre vos instructions thoroughly. Question: Does this imply that Un member de mon team exécutera vos ordres avec une grande précision ? Yes, no, or maybe?\n Output: yes\nGays et lesbians . Question: Does this imply that Les heterosexuals? Yes, no, or maybe?\n Output: no\nJ' ai traversé un together de doors de cabine , et je suis tombé au floor. Question: Does this imply that J' ai crossed les doors et je suis fallen . ? Yes, no, or maybe?\n Output: yes\n Amusant pour les adultes et les children .Question: Does this imply that Amusant pour seulement les children . ? Yes, no, or maybe?\n Output: no\nIl s' est returned et a smiled à vrenna . Question: Does this imply that Il souritt à vrenna qui marchait slowly derrière lui avec sa mother . ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_fr_80p = f"L' skimming conceptuel de la crème a deux dimensions fondamentales : le product et la geography. Question: Does this imply that Le product et la geography sont ce qui fait travailler la crème de la crème? Yes, no, or maybe?\n Output: maybe\nUn de nos numéros vous will follow vos instructions thoroughly. Question: Does this imply that Un member de mon team exécutera vos ordres avec une grande accuracy ? Yes, no, or maybe?\n Output: yes\nGays et lesbians . Question: Does this imply that Les heterosexuals? Yes, no, or maybe?\n Output: no\nJ' ai crossedun together de doors de cabin , et je suis fallen au floor. Question: Does this imply that J' ai crossed les doors et je suis fallen . ? Yes, no, or maybe?\n Output: yes\n Amusant pour les adult et les children .Question: Does this imply that Amusant pour seulement les children . ? Yes, no, or maybe?\n Output: no\nIl s' est returned et a smiled à vrenna . Question: Does this imply that Il sourit à vrenna qui marchait slowly derrière lui avec sa mother . ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_de = f"Konzeptionelles Cream-Skimming hat zwei grundlegende Dimensionen: Produkt und Geografie. Question: Does this imply that Sind es Produkt und Geografie, die dafür sorgen, dass die Crème de la Crème funktioniert? Yes, no, or maybe?\n Output: maybe\nUnter einer unserer Nummern werden Sie aufgefordert, Ihre Anweisungen sorgfältig zu befolgen. Question: Does this imply that Ein Mitglied meines Teams wird Ihre Aufträge mit großer Präzision ausführen ? Yes, no, or maybe?\n Output: yes\nSchwule und Lesben. Question: Does this imply that Heterosexuelle? Yes, no, or maybe?\n Output: no\nIch ging durch eine Kabinentür und fiel zu Boden. Question: Does this imply that Ich ging durch die Türen und fiel. ? Yes, no, or maybe?\n Output: yes\n Spaß für Erwachsene und Kinder. Question: Does this imply that Spaß nur für Kinder. ? Yes, no, or maybe?\n Output: no\nEr drehte sich um und lächelte Vrenna an. Question: Does this imply that Er lächelte Vrenna an, die mit ihrer Mutter langsam hinter ihm herging. ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_de_30p = f"Konzeptionelles Cream-Skimming hat zwei basic Dimensionen: Produkt und Geografie. Question: Does this imply that Sind es Produkt und Geografie, die dafür worry, dass die Crème de la Crème funktioniert? Yes, no, or maybe?\n Output: maybe\nUnter einer unserer Nummern werden Sie urged, Ihre Anweisungen sorgfältig zu befolgen. Question: Does this imply that Ein member meines Teams wird Ihre Aufträge mit großer Präzision ausführen ? Yes, no, or maybe?\n Output: yes\nGays und Lesben. Question: Does this imply that Heterosexuelle? Yes, no, or maybe?\n Output: no\nIch ging durch eine Kabinentür und fiel zu Boden. Question: Does this imply that Ich walked durch die Türen und fiel. ? Yes, no, or maybe?\n Output: yes\n Spaß für Erwachsene und Kinder. Question: Does this imply that Spaß nur für Kinder. ? Yes, no, or maybe?\n Output: no\nEr drehte sich um und lächelte Vrenna an. Question: Does this imply that Er lächelte Vrenna an, die mit ihrer mother langsam hinter ihm herging. ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_de_50p = f"Konzeptionelles Cream-Skimming hat zwei basic dimension: products und Geografie. Question: Does this imply that Sind es products und Geografie, die dafür worry, dass die Crème de la Crème funktioniert? Yes, no, or maybe?\n Output: maybe\nUnter einer unserer numbers werden Sie urged, Ihre Anweisungen sorgfältig zu befolgen. Question: Does this imply that Ein member meines Teams wird Ihre orders mit großer Präzision ausführen ? Yes, no, or maybe?\n Output: yes\nGays und lesbians. Question: Does this imply that Heterosexuals? Yes, no, or maybe?\n Output: no\nIch ging durch eine Kabinentür und fiel zu floor. Question: Does this imply that Ich walked durch die Türen und dropped. ? Yes, no, or maybe?\n Output: yes\n Kidding für Erwachsene und Kinder. Question: Does this imply that Kidding nur für Kinder. ? Yes, no, or maybe?\n Output: no\nEr drehte sich um und lächelte Vrenna an. Question: Does this imply that Er lächelte Vrenna an, die mit ihrer mother langsam hinter ihm herging. ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    prompt_de_80p = f"Konzeptionelles Cream-Skimming hat zwei basic dimension: products und geography. Question: Does this imply that Sind es products und geography, die dafür worry, dass die Crème de la Crème works? Yes, no, or maybe?\n Output: maybe\nUnter einer unserer numbers werden Sie urged, Ihre Anweisungen sorgfältig zu obey. Question: Does this imply that Ein member meines Teams wird Ihre orders mit großer accuracy executes ? Yes, no, or maybe?\n Output: yes\nGays und lesbians. Question: Does this imply that Heterosexuals? Yes, no, or maybe?\n Output: no\nIch walked durch eine Kabinentür und dropped zu floor. Question: Does this imply that Ich walked durch die doors und dropped. ? Yes, no, or maybe?\n Output: yes\n Kidding für Erwachsene und children. Question: Does this imply that Kidding just für Kinder. ? Yes, no, or maybe?\n Output: no\nEr rotated sich um und lächelte Vrenna an. Question: Does this imply that Er lächelte Vrenna an, die mit ihrer mother slowly hinter ihm herging. ? Yes, no, or maybe?\n Output: maybe\n{premise} Question: Does this imply that {hypothesis} ?Yes, no, or maybe?\n Output: "
    
    # temp = sys.argv[1]
    # prompt = locals().get(temp, "Variable not found!")
    prompt = prompt_eng
    input_ids = tokenizer(prompt, return_tensors="pt").to(0)
    sample = model.generate(**input_ids, max_length=1500)
    output = tokenizer.decode(sample[0]).split("<pad>")[1].split("</s>")[0]
    output = output.split()[0]
    # print(output, correct_label)
    

    prediction = output_map_default[output.strip().lower()]
    
    predictions.append(prediction)
    ground_truth.append(correct_label)
    resultList.append([premise, hypothesis, prediction, correct_label])

    if count % 2500 == 0:
        print("{} instances done".format(count))


import pandas as pd

# ... [rest of your code]

# Compute overall (macro) metrics
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


filename = f'MT0_xxl_results/result_hi_xnli_st'
fields = ['premise', 'hypothesis', 'prediction', 'correct_label']

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(resultList)
