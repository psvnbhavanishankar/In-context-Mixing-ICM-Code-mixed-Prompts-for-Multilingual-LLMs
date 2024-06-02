import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed, AutoModelForCausalLM
import os
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # SET the GPUs you want to use
import sys

# Following line makes the model load in memory
torch.set_default_tensor_type(torch.cuda.FloatTensor)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "models/mt0-xxl-mt")
tokenizer = AutoTokenizer.from_pretrained("models/mt0-xxl-mt")

# model = AutoModelForCausalLM.from_pretrained(
#     "bigscience/bloomz-3b", use_cache=True)
# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-3b")

# model = AutoModelForCausalLM.from_pretrained(
#     "facebook/xglm-2.9B", use_cache=True
# tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-2.9B")

# Following line for reproducibility
set_seed(2023)
# set_seed(42)

# temp = sys.argv[1]
# f_open = locals().get(temp, "Variable not found!")

testFile = open(sys.argv[1], 'r')
reader = csv.reader(testFile)
testList = []
for instance in reader:
    testList.append(instance)

resultList = []
count = 0

for i in testList[1:]:

    count += 1
    instance = []
    instance.append(i[0])
    instance.append(i[1])

    text_input = i[0]
    
    # prompt = f"Disfluencies are the phenomenon which break the conversation flow of a sentence. Disfluencies include fillers, pauses, interjections, discourse markers, repetitions, corrections, edits and false starts. Remove disfluencies to produce a fluent sentences. Input: I umm.. don't want to go uhh.. to the school. Remove disfluencies in the given sentence.\nOutput: I don't want to go to the school.\nInput: The sun sun rises in the east. Remove disfluencies in the given sentence.\nOutput: The sun rises in the east.\nInput: There are nin eight planets in the solar system. Remove disfluencies in the given sentence.\nOutput: There are eight planets in the solar system.\nInput: Let's meet at 2 sorry 3. Remove disfluencies in the given sentence.\nOutput: Let's meet at 3.\nInput: Ouch! It hurts.  Remove disfluencies in the given sentence.\nOutput: It hurts.\nInput: Well, I don't think I should try this. Remove disfluencies in the given sentence.\nOutput: I don't think I should try this.\nInput: Saketh and I drove I'm hungry now is there something to eat? Remove disfluencies in the given sentence.\nOutput: I'm hungry now is there something to eat?\nInput: I went to the market today. Remove disfluencies in the given sentence.\nOutput: I went to the market today.\nInput: {text_input}\nOutput: "
    prompt_eng = f"Input: I umm.. don't want to go uhh.. to the school. Remove disfluencies in the given sentence.\nOutput: I don't want to go to the school.\nInput: The sun sun rises in the east. Remove disfluencies in the given sentence.\nOutput: The sun rises in the east.\nInput: There are nin eight planets in the solar system. Remove disfluencies in the given sentence.\nOutput: There are eight planets in the solar system.\nInput: Let's meet at 2 sorry 3. Remove disfluencies in the given sentence.\nOutput: Let's meet at 3.\nInput: Ouch! It hurts.  Remove disfluencies in the given sentence.\nOutput: It hurts.\nInput: Well, I don't think I should try this. Remove disfluencies in the given sentence.\nOutput: I don't think I should try this.\nInput: Saketh and I drove I'm hungry now is there something to eat? Remove disfluencies in the given sentence.\nOutput: I'm hungry now is there something to eat?\nInput: I went to the market today. Remove disfluencies in the given sentence.\nOutput: I went to the market today.\nInput: {text_input} Remove disfluencies in the given sentence. Ensure not to give any output in English. Output only in the language of the input.\nOutput: "
    # union_prompt =f"Input: I umm.. don't want to go uhh.. to the school. Remove disfluencies in the given sentence.\nOutput: I don't want to go to the school.\nInput: The sun sun rises in the east. Remove disfluencies in the given sentence.\nOutput: The sun rises in the east.\nInput: There are nin eight planets in the solar system. Remove disfluencies in the given sentence.\nOutput: There are eight planets in the solar system.\nInput: Let's meet at 2 sorry 3. Remove disfluencies in the given sentence.\nOutput: Let's meet at 3.\nInput: Ouch! It hurts.  Remove disfluencies in the given sentence.\nOutput: It hurts.\nInput: Well, I don't think I should try this. Remove disfluencies in the given sentence.\nOutput: I don't think I should try this.\nInput: Saketh and I drove I'm hungry now is there something to eat? Remove disfluencies in the given sentence.\nOutput: I'm hungry now is there something to eat?\nInput: I went to the market today. Remove disfluencies in the given sentence.\nOutput: I went to the market today.\nInput: నాకూ ఉమ్మ్ ఇడ్లి తినాలని ఉంది. Remove disfluencies in the given sentence.\nOutput: నాకూ ఇడ్లి తినాలని ఉంది.\nInput: రాధ రాధ అనుకుంటా ఈసారి టాపర్. Remove disfluencies in the given sentence.\nOutput: రాధ అనుకుంటా ఈసారి టాపర్.\nInput: సూర్యుని చుట్టూ తొమ్మి ఎనిమిది గ్రహాలు తిరుగుతున్నాయి. Remove disfluencies in the given sentence.\nOutput: సూర్యుని చుట్టూ ఎనిమిది గ్రహాలు తిరుగుతున్నాయి.\nInput: రెండు కి సారీ మూడుకు కలుద్దాం. Remove disfluencies in the given sentence.\nOutput: మూడుకు కలుద్దాం.\nInput: ఒరేయ్ నొప్పి గా ఉంది. Remove disfluencies in the given sentence.\nOutput: నొప్పి గా ఉంది.\nInput: సర్లే కాని పని పూర్తి చెయ్యి నువ్వు ముందు. Remove disfluencies in the given sentence.\nOutput: పని పూర్తి చెయ్యి నువ్వు ముందు.\nInput: సాకేత్ నేను కలిసి చాలా ఆకలిగా ఉంది తినడానికి ఏమైనా దొరుకుతుందా? Remove disfluencies in the given sentence.\nOutput: చాలా ఆకలిగా ఉంది తినడానికి ఏమైనా దొరుకుతుందా?\nInput: ఈరోజు నేను బజారుకు వెళ్లాను. Remove disfluencies in the given sentence.\nOutput: ఈరోజు నేను బజారుకు వెళ్లాను.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    # telugu_prompt =f"Input: నాకూ ఉమ్మ్ ఇడ్లి తినాలని ఉంది. Remove disfluencies in the given sentence.\nOutput: నాకూ ఇడ్లి తినాలని ఉంది.\nInput: రాధ రాధ అనుకుంటా ఈసారి టాపర్. Remove disfluencies in the given sentence.\nOutput: రాధ అనుకుంటా ఈసారి టాపర్.\nInput: సూర్యుని చుట్టూ తొమ్మి ఎనిమిది గ్రహాలు తిరుగుతున్నాయి. Remove disfluencies in the given sentence.\nOutput: సూర్యుని చుట్టూ ఎనిమిది గ్రహాలు తిరుగుతున్నాయి.\nInput: రెండు కి సారీ మూడుకు కలుద్దాం. Remove disfluencies in the given sentence.\nOutput: మూడుకు కలుద్దాం.\nInput: ఒరేయ్ నొప్పి గా ఉంది. Remove disfluencies in the given sentence.\nOutput: నొప్పి గా ఉంది.\nInput: సర్లే కాని పని పూర్తి చెయ్యి నువ్వు ముందు. Remove disfluencies in the given sentence.\nOutput: పని పూర్తి చెయ్యి నువ్వు ముందు.\nInput: సాకేత్ నేను కలిసి చాలా ఆకలిగా ఉంది తినడానికి ఏమైనా దొరుకుతుందా? Remove disfluencies in the given sentence.\nOutput: చాలా ఆకలిగా ఉంది తినడానికి ఏమైనా దొరుకుతుందా?\nInput: ఈరోజు నేను బజారుకు వెళ్లాను. Remove disfluencies in the given sentence.\nOutput: ఈరోజు నేను బజారుకు వెళ్లాను.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    # telugu_prompt_long =f"Input: నేను ఉహుహు... ప్రపంచ కప్ మ్యాచ్ చూడాలనుకుంటున్నాను. Remove disfluencies in the given sentence.\nOutput: నేను ప్రపంచ కప్ మ్యాచ్ చూడాలనుకుంటున్నాను.\nInput: అరేరే నేను నా మొబైల్ ఎక్కడ పెట్టానో మరచిపోయాను! Remove disfluencies in the given sentence.\nOutput: నేను నా మొబైల్ ఎక్కడ పెట్టానో మరచిపోయాను!\nInput: ఆయన బాగా బాగా ఆడుతున్నాడు క్రికెట్. Remove disfluencies in the given sentence.\nOutput: ఆయన బాగా ఆడుతున్నాడు క్రికెట్.\nInput: మన ప్రయాణం శుక్రవా శనివారం నాడు కాదండి? Remove disfluencies in the given sentence.\nOutput: మన ప్రయాణం శనివారం నాడు కాదండి?\nInput: 3:30 కాదు కాదు, 4:30 కి మన ప్రయాణం. Remove disfluencies in the given sentence.\nOutput: 4:30 కి మన ప్రయాణం.\nInput: నా పుస్తకం నాకు పేపర్ కావాలి. Remove disfluencies in the given sentence.\nOutput: నాకు పేపర్ కావాలి.\nInput: నాకూ ఉమ్మ్ ఇడ్లి తినాలని ఉంది. Remove disfluencies in the given sentence.\nOutput: నాకూ ఇడ్లి తినాలని ఉంది.\nInput: రాధ రాధ అనుకుంటా ఈసారి టాపర్. Remove disfluencies in the given sentence.\nOutput: రాధ అనుకుంటా ఈసారి టాపర్.\nInput: సూర్యుని చుట్టూ తొమ్మి ఎనిమిది గ్రహాలు తిరుగుతున్నాయి. Remove disfluencies in the given sentence.\nOutput: సూర్యుని చుట్టూ ఎనిమిది గ్రహాలు తిరుగుతున్నాయి.\nInput: రెండు కి సారీ మూడుకు కలుద్దాం. Remove disfluencies in the given sentence.\nOutput: మూడుకు కలుద్దాం.\nInput: ఒరేయ్ నొప్పి గా ఉంది. Remove disfluencies in the given sentence.\nOutput: నొప్పి గా ఉంది.\nInput: సర్లే కాని పని పూర్తి చెయ్యి నువ్వు ముందు. Remove disfluencies in the given sentence.\nOutput: పని పూర్తి చెయ్యి నువ్వు ముందు.\nInput: సాకేత్ నేను కలిసి చాలా ఆకలిగా ఉంది తినడానికి ఏమైనా దొరుకుతుందా? Remove disfluencies in the given sentence.\nOutput: చాలా ఆకలిగా ఉంది తినడానికి ఏమైనా దొరుకుతుందా?\nInput: ఈరోజు నేను బజారుకు వెళ్లాను. Remove disfluencies in the given sentence.\nOutput: ఈరోజు నేను బజారుకు వెళ్లాను.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    # mixed_prompt = f"Input: ఇది 21st podcast ఇందులో only two guest two female guestలని తీస్కరాగలిగాము మేము. Remove disfluencies in the given sentence.\nOutput: ఇది 21st podcast ఇందులో only two female guestలని తీస్కరాగలిగాము మేము.\nInput: and మా approach కూడా outreach కూడా కొంచెం ఉహుహు.. uhh speed up చేసాము మేము. Remove disfluencies in the given sentence.\nOutput: and మా approach కూడా outreach కూడా కొంచెం speed up చేసాము మేము.\nInput: because ఎందుకంటే మా filtration process కొంచెం differentగా ఉంటది. Remove disfluencies in the given sentence.\nOutput: ఎందుకంటే మా filtration process కొంచెం differentగా ఉంటది.\nInput: uhh so infact like మీ మీ మా మాట్లాడే విధానం ఏదైతే ఉందో. Remove disfluencies in the given sentence.\nOutput: infact మీ మాట్లాడే విధానం ఏదైతే ఉందో.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    # mixed_30p_prompt = f"Input:నేను ఉహుహు... ప్రపంచ కప్ Match చూడాలనుకుంటున్నాను. Remove disfluencies in the given sentence.\nOutput: నేను ప్రపంచ కప్ Match చూడాలనుకుంటున్నాను.\nInput: అరేరే నేను నా మొబైల్ ఎక్కడ put మరచిపోయాను! Remove disfluencies in the given sentence.\nOutput: నేను నా మొబైల్ ఎక్కడ put మరచిపోయాను!\nInput: ఆయన బాగా బాగా ఆడుతున్నాడు క్రికెట్. Remove disfluencies in the given sentence.\nOutput: ఆయన బాగా ఆడుతున్నాడు క్రికెట్.\nInput: మన ప్రయాణం శుక్రవా శనివారం day కాదండి? Remove disfluencies in the given sentence.\nOutput: మన ప్రయాణం శనివారం day కాదండి?\nInput: 3:30 కాదు కాదు, 4:30 కి మన travel. Remove disfluencies in the given sentence.\nOutput: 4:30 కి మన travel. Remove disfluencies in the given sentence.\nInput: నా book నాకు పేపర్ కావాలి.\nOutput: నాకు పేపర్ కావాలి.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    # mixed_50p_prompt = f"Input:నేను ఉహుహు... World కప్ Match చూడాలనుకుంటున్నాను. Remove disfluencies in the given sentence.\nOutput: నేను World కప్ Match చూడాలనుకుంటున్నాను.\nInput: అరేరే నేను నా mobile ఎక్కడ put మరచిపోయాను! Remove disfluencies in the given sentence.\nOutput: నేను నా mobile ఎక్కడ put మరచిపోయాను!\nInput: ఆయన బాగా బాగా ఆడుతున్నాడు cricket. Remove disfluencies in the given sentence.\nOutput: ఆయన బాగా ఆడుతున్నాడు cricket.\nInput: మన travel శుక్రవా శనివారం day కాదండి? Remove disfluencies in the given sentence.\nOutput: మన travel శనివారం day కాదండి?\nInput: 3:30 కాదు no, 4:30 కి మన travel. Remove disfluencies in the given sentence.\nOutput: 4:30 కి మన travel.\nInput: నా book నాకు paper కావాలి. Remove disfluencies in the given sentence.\nOutput: నాకు paper కావాలి.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    # mixed_80p_prompt = f"Input:నేను ఉహుహు... World cup Match చూడాలనుకుంటున్నాను. Remove disfluencies in the given sentence.\nOutput: నేను World cup Match చూడాలనుకుంటున్నాను.\nInput: అరేరే నేను my mobile ఎక్కడ put మరచిపోయాను! Remove disfluencies in the given sentence.\nOutput: నేను my mobile ఎక్కడ put మరచిపోయాను!\nInput: ఆయన very very ఆడుతున్నాడు క్రికెట్. Remove disfluencies in the given sentence.\nOutput: ఆయన very ఆడుతున్నాడు cricket.\nInput: Our travel శుక్రవా శనివారం day కాదండి? Remove disfluencies in the given sentence.\nOutput: Our travel శనివారం day కాదండి?\nInput: 3:30 no no, 4:30 కి మన travel. Remove disfluencies in the given sentence.\nOutput: 4:30 కి మన travel.\nInput: నా book నాకు paper need. Remove disfluencies in the given sentence.\nOutput: నాకు paper need.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_prompt = f"Input: अ मुझे बताइए ये सेवा नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे बताइए ये सेवा नहीं है क्या?\nInput: लेकिन यहां पर पर अटकने से काम नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन यहां पर अटकने से काम नहीं होगा।\nInput: बाजपेयी भी इन अर अर्थशास्त्रियों में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी भी इन अर्थशास्त्रियों में शामिल थे।\nInput: मैंने सोचा कि, उम्म...क्या मैं आज शाम को moview देखने जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज शाम को फिल्म देखने जाऊँ?\nInput: अरे, यह कुत्ता हमारे पास क्यों आ रहा है? Remove disfluencies in the given sentence.\nOutput: यह कुत्ता हमारे पास क्यों आ रहा है?\nInput: क्या हमें कल.. हमें कल चलना चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल चलना चाहिए।\nInput: उसके...मतलब, उसने अपनी नई कार खरीदी है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी नई कार खरीदी है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_30p = f"Input: अ मुझे बताइए ये service नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे बताइए ये service नहीं है क्या?\nInput: लेकिन यहां पर पर अटकने से work नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन यहां पर अटकने से work नहीं होगा।\nInput: बाजपेयी also इन अर अर्थशास्त्रियों में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी also इन अर्थशास्त्रियों में शामिल थे।\nInput: मैंने सोचा कि, उम्म...क्या मैं आज शाम को फिल्म see जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज शाम को फिल्म see जाऊँ?\nInput: अरे, यह कुत्ता हमारे near क्यों आ रहा है? Remove disfluencies in the given sentence.\nOutput: यह कुत्ता हमारे near क्यों आ रहा है?\nInput: क्या हमें कल.. हमें कल walk चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल walk चाहिए।\nInput: उसके...मतलब, उसने अपनी नई car खरीदी है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी नई car खरीदी है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_50p = f"Input: अ मुझे बताइए ये service नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे बताइए ये service नहीं है क्या?\nInput: लेकिन here पर पर अटकने से work नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन here पर अटकने से work नहीं होगा।\nInput: बाजपेयी भी इन अर economists में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी भी इन economists में शामिल थे।\nInput: मैंने सोचा कि, उम्म...क्या मैं आज evening को film देखने जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज evening को film देखने जाऊँ?\nInput: अरे, यह dog हमारे near क्यों आ रहा है? Remove disfluencies in the given sentence.\nOutput: यह dog हमारे near क्यों आ रहा है?\nInput: क्या हमें कल.. हमें कल go चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल go चाहिए।\nInput: उसके...मतलब, उसने अपनी new car purchase है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी new car purchase है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_80p = f"Input: अ मुझे tell ये service नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे tell ये service नहीं है क्या?\nInput: लेकिन here पर पर cling से work नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन here पर cling से work नहीं होगा।\nInput: बाजपेयी also इन अर economists में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी also इन economists में शामिल थे।\nInput: मैंने thought कि, उम्म...क्या मैं आज evening को film see जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज evening को film see जाऊँ?\nInput: अरे, यह dog हमारे near क्यों come रहा है? Remove disfluencies in the given sentence.\nOutput: यह dog हमारे near क्यों come रहा है?\nInput: क्या हमें tomorrow.. हमें कल walk चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल walk चाहिए।\nInput: उसके...मतलब, उसने अपनी new car bought है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी new car bought है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_30pf = f"Input: अ मुझे बताइए ये service नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे बताइए ये service नहीं है क्या?\nInput: लेकिन यहां पर पर अटकने से travail नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन यहां पर अटकने से travail नहीं होगा।\nInput: बाजपेयी aussi इन अर अर्थशास्त्रियों में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी aussi इन अर्थशास्त्रियों में शामिल थे।\nInput: मैंने सोचा कि, उम्म...क्या मैं आज शाम को फिल्म voir जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज शाम को फिल्म voir जाऊँ?\nInput: अरे, यह कुत्ता हमारे près क्यों आ रहा है? Remove disfluencies in the given sentence.\nOutput: यह कुत्ता हमारे près क्यों आ रहा है?\nInput: क्या हमें कल.. हमें कल marcher चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल marcher चाहिए।\nInput: उसके...मतलब, उसने अपनी नई voiture खरीदी है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी नई voiture खरीदी है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_50pf = f"Input: अ मुझे बताइए ये service नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे बताइए ये service नहीं है क्या?\nInput: लेकिन here पर पर अटकने से travail नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन here पर अटकने से travail नहीं होगा।\nInput: बाजपेयी भी इन अर économistes में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी भी इन économistes में शामिल थे।\nInput: मैंने सोचा कि, उम्म...क्या मैं आज soir को film देखने जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज soir को film देखने जाऊँ?\nInput: अरे, यह chien हमारे près क्यों आ रहा है? Remove disfluencies in the given sentence.\nOutput: यह chien हमारे près क्यों आ रहा है?\nInput: क्या हमें कल.. हमें कल aller चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल aller चाहिए।\nInput: उसके...मतलब, उसने अपनी Nouveau voiture achat है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी Nouveau voiture achat है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_80pf = f"Input: अ मुझे dire ये service नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे dire ये service नहीं है क्या?\nInput: लेकिन here पर पर bloqué से travail नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन here पर bloqué से travail नहीं होगा।\nInput: बाजपेयी aussi इन अर économistes में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी aussi इन économistes में शामिल थे।\nInput: मैंने pensé कि, उम्म...क्या मैं आज soir को film see जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज soir को film see जाऊँ?\nInput: अरे, यह chien हमारे près क्यों viens रहा है? Remove disfluencies in the given sentence.\nOutput: यह chien हमारे près क्यों viens रहा है?\nInput: क्या हमें tomorrow.. हमें कल marcher चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल marcher चाहिए।\nInput: उसके...मतलब, उसने अपनी Nouveau voiture acheté है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी Nouveau voiture acheté है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    marathi_prompt = f"Input: अ मला सांगा, ही सेवा नाही का? Remove disfluencies in the given sentence.\nOutput: मला सांगा, ही सेवा नाही का?\nInput: माझ्या माझ्या कामाची चर्चा आहे उद्या. Remove disfluencies in the given sentence.\nOutput: माझ्या कामाची चर्चा आहे उद्या.\nInput: या अर अर्थतज्ज्ञांमध्ये वाजपेयींचाही समावेश होता. Remove disfluencies in the given sentence.\nOutput: या अर्थतज्ज्ञांमध्ये वाजपेयींचाही समावेश होता.\nInput: मी विचार केला कि, अं...मी आज सायंकाळी मी चित्रपट पहायला जाऊ का? Remove disfluencies in the given sentence.\nOutput: आज सायंकाळी मी चित्रपट पहायला जाऊ का?\nInput: अरे हा कुत्रा आमच्याजवळ का येतोय? Remove disfluencies in the given sentence.\nOutput: हा कुत्रा आमच्याजवळ का येतोय?\nInput: का आपण उद्या.. आपण उद्या जायला हवं. Remove disfluencies in the given sentence.\nOutput: आपण उद्या जायला हवं.\nInput: त्याची...म्हणजे, त्याने त्याची नवीन कार खरेदी केली आहे. Remove disfluencies in the given sentence.\nOutput: त्याने त्याची नवीन कार खरेदी केली आहे.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    marathi_30p = f"Input: अ मला सांगा, ही service नाही का? Remove disfluencies in the given sentence.\nOutput: मला सांगा, ही service नाही का?\nInput: माझ्या माझ्या work चर्चा आहे उद्या. Remove disfluencies in the given sentence.\nOutput: माझ्या work चर्चा आहे उद्या.\nInput: या अर अर्थतज्ज्ञांमध्ये वाजपेयींचाही समावेश होता. Remove disfluencies in the given sentence.\nOutput: या अर्थतज्ज्ञांमध्ये वाजपेयींचाही समावेश होता.\nInput: मी विचार केला कि, अं...मी आज सायंकाळी मी film पहायला जाऊ का? Remove disfluencies in the given sentence.\nOutput: आज सायंकाळी मी film पहायला जाऊ का?\nInput: अरे हा कुत्रा आमच्याजवळ का येतोय? Remove disfluencies in the given sentence.\nOutput: हा कुत्रा आमच्याजवळ का येतोय?\nInput: का आपण उद्या.. आपण उद्या go हवं. Remove disfluencies in the given sentence.\nOutput: आपण उद्या go हवं.\nInput: त्याची...म्हणजे, त्याने त्याची new कार खरेदी केली आहे. Remove disfluencies in the given sentence.\nOutput: त्याने त्याची new कार खरेदी केली आहे.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    marathi_50p = f"Input: अ मला सांगा, ही service नाही का? Remove disfluencies in the given sentence.\nOutput: मला सांगा, ही service नाही का?\nInput: माझ्या माझ्या work discussion आहे उद्या. Remove disfluencies in the given sentence.\nOutput: माझ्या कामाची work discussion आहे उद्या.\nInput: या अर economists ये वाजपेयींचाही समावेश होता. Remove disfluencies in the given sentence.\nOutput: या economists वाजपेयींचाही समावेश होता.\nInput: मी विचार केला कि, अं...मी आज सायंकाळी मी film see जाऊ का? Remove disfluencies in the given sentence.\nOutput: आज सायंकाळी मी film seeजाऊ का?\nInput: अरे हा dog आमच्याजवळ का येतोय? Remove disfluencies in the given sentence.\nOutput: हा dog आमच्याजवळ का येतोय?\nInput: का आपण उद्या.. आपण tomorrow go हवं. Remove disfluencies in the given sentence.\nOutput: आपण tomorrow go हवं.\nInput: त्याची...म्हणजे, त्याने त्याची new car खरेदी केली आहे. Remove disfluencies in the given sentence.\nOutput: त्याने त्याची new car खरेदी केली आहे.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    bengali_prompt = f"Input: আহ আমাকে বলুন, এটা কি পরিষেবা নয়? Remove disfluencies in the given sentence.\nOutput: আমাকে বলুন, এটা কি পরিষেবা নয়?\nInput: কিন্তু কিন্তু এখানে আটকালে তো কাজ হবেনা। Remove disfluencies in the given sentence.\nOutput: কিন্তু এখানে আটকালে তো কাজ হবেনা।\nInput: বাজপেয়ী জিও এইসব অর্থ অর্থনীতি তে অন্তর্ভুক্ত ছিলেন। Remove disfluencies in the given sentence.\nOutput: বাজপেয়ী জিও এইসব অর্থনীতি তে অন্তর্ভুক্ত ছিলেন।\nInput: আমি ভাবলাম কি যে, আঃ আজকে সন্ধায় কি আমি সিনেমা দেখতে যাব? Remove disfluencies in the given sentence.\nOutput: আজকে সন্ধায় কি আমি সিনেমা দেখতে যাব?\nInput: আরেহ এই কুকুর টা আমাদের দিকে কেন আসছে? Remove disfluencies in the given sentence.\nOutput: এই কুকুর টা আমাদের দিকে কেন আসছে?\nInput: আমিাদের কি কাল, আমাদের কাল যাওয়া উচিত। Remove disfluencies in the given sentence.\nOutput: আমাদের কাল যাওয়া উচিত।\nInput: त्ওনার, মানে উনি নিজের নতুন গাড়ি কিনেছেন। Remove disfluencies in the given sentence.\nOutput: त्উনি নিজের নতুন গাড়ি কিনেছেন।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    bengali_30p = f"Input: আহ আমাকে বলুন, এটা কি service নয়? Remove disfluencies in the given sentence.\nOutput: আমাকে বলুন, এটা কি service নয়?\nInput: কিন্তু কিন্তু এখানে আটকালে তো work হবেনা। Remove disfluencies in the given sentence.\nOutput: কিন্তু এখানে আটকালে তো work হবেনা।\nInput: বাজপেয়ী জিও এইসব অর্থ অর্থনীতি তে অন্তর্ভুক্ত ছিলেন। Remove disfluencies in the given sentence.\nOutput: বাজপেয়ী জিও এইসব অর্থনীতি তে অন্তর্ভুক্ত ছিলেন।\nInput: আমি ভাবলাম কি যে, আঃ আজকে সন্ধায় কি আমি সিনেমা see যাব? Remove disfluencies in the given sentence.\nOutput: আজকে সন্ধায় কি আমি সিনেমা see যাব?\nInput: আরেহ এই dog টা আমাদের দিকে কেন আসছে? Remove disfluencies in the given sentence.\nOutput: এই dog টা আমাদের দিকে কেন আসছে?\nInput: আমিাদের কি কাল, আমাদের কাল walk উচিত। Remove disfluencies in the given sentence.\nOutput: আমাদের কাল walk উচিত।\nInput: त्ওনার, মানে উনি নিজের নতুন car কিনেছেন। Remove disfluencies in the given sentence.\nOutput: त्উনি নিজের নতুন car কিনেছেন।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    bengali_50p = f"Input: আহ আমাকে বলুন, এটা কি service নয়? Remove disfluencies in the given sentence.\nOutput: আমাকে বলুন, এটা কি service নয়?\nInput: কিন্তু কিন্তু here আটকালে তো work হবেনা। Remove disfluencies in the given sentence.\nOutput: কিন্তু here আটকালে তো work হবেনা।\nInput: বাজপেয়ী জিও এইসব অর্থ economists তে অন্তর্ভুক্ত ছিলেন। Remove disfluencies in the given sentence.\nOutput: বাজপেয়ী জিও এইসব economists তে অন্তর্ভুক্ত ছিলেন।\nInput: আমি ভাবলাম কি যে, আঃ আজকে evening কি আমি film দেখতে যাব? Remove disfluencies in the given sentence.\nOutput: আজকে evening কি আমি film দেখতে যাব?\nInput: আরেহ এই dog টা আমাদের near কেন আসছে? Remove disfluencies in the given sentence.\nOutput: এই dog টা আমাদের near কেন আসছে?\nInput: আমিাদের কি কাল, আমাদের কাল go উচিত। Remove disfluencies in the given sentence.\nOutput: আমাদের কাল go উচিত।\nInput: त्ওনার, মানে উনি নিজের new car কিনেছেন। Remove disfluencies in the given sentence.\nOutput: त्উনি নিজের new car কিনেছেন।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    bengali_80p = f"Input: আহ আমাকে tell, এটা কি service নয়? Remove disfluencies in the given sentence.\nOutput: আমাকে tell, এটা কি service নয়?\nInput: কিন্তু কিন্তু here cling তো work হবেনা। Remove disfluencies in the given sentence.\nOutput: কিন্তু here cling তো work হবেনা।\nInput: বাজপেয়ী জিও এইসব অর্থ economists তে include ছিলেন। Remove disfluencies in the given sentence.\nOutput: বাজপেয়ী জিও এইসব economists তে includeত ছিলেন।\nInput: আমি thought কি যে, আঃ আজকে evening কি আমি film দেখতে যাব? Remove disfluencies in the given sentence.\nOutput: আজকে evening কি আমি film দেখতে যাব?\nInput: আরেহ এই dog টা আমাদের near কেন coming? Remove disfluencies in the given sentence.\nOutput: এই dog টা আমাদের near কেন coming?\nInput: আমিাদের কি tomorrow, আমাদের কাল walk উচিত। Remove disfluencies in the given sentence.\nOutput: আমাদের কাল walk উচিত।\nInput: त्ওনার, মানে উনি নিজের new car purchase। Remove disfluencies in the given sentence.\nOutput: त्উনি নিজের new car purchase।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    telugu_translation = f"Input: I umm.. want to watch the World Cup match. Remove disfluencies in the given sentence.\nOutput: I want to watch the World Cup match.\nInput: Oh, I forgot where I put my mobile! Remove disfluencies in the given sentence.\nOutput: I forgot where I put my mobile!\nInput: He is playing playing cricket very well. Remove disfluencies in the given sentence.\nOutput: He is playing cricket very well.\nInput: Our trip is on Fri Saturday, isn't it? Remove disfluencies in the given sentence.\nOutput: Our trip is on Saturday, isn't it?\nInput: Our journey is at 3:30 no no 4:30. Remove disfluencies in the given sentence.\nOutput: Our journey is at 4:30.\nInput: my book I need a paper. Remove disfluencies in the given sentence.\nOutput: I need a paper.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    bengali_translation = f"Input: uhh Tell me, is this not a service? Remove disfluencies in the given sentence.\nOutput: Tell me, is this not a service?\nInput: But but it will not work by staying here. Remove disfluencies in the given sentence.\nOutput: But it will not work by staying here.\nInput: Vajpayee was also among these  econ economists. Remove disfluencies in the given sentence.\nOutput: Vajpayee was also among these economists.\nInput: I thought, um...maybe I'll go see a movie tonight? Remove disfluencies in the given sentence.\nOutput: maybe I'll go see a movie tonight?\nInput: Hey, why is this dog coming to us? Remove disfluencies in the given sentence.\nOutput: why is this dog coming to us?\nInput: should we we should go tomorrow. Remove disfluencies in the given sentence.\nOutput: we should go tomorrow.\nInput: His, I mean he bought a new car. Remove disfluencies in the given sentence.\nOutput: he bought a new car.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_translation = f"Input: uhh Tell me, is this not a service? Remove disfluencies in the given sentence.\nOutput: Tell me, is this not a service?\nInput: But but it will not work by staying here. Remove disfluencies in the given sentence.\nOutput: But it will not work by staying here.\nInput: Vajpayee was also among these  econ economists. Remove disfluencies in the given sentence.\nOutput: Vajpayee was also among these economists.\nInput: I thought, um...maybe I'll go see a movie tonight? Remove disfluencies in the given sentence.\nOutput: maybe I'll go see a movie tonight?\nInput: Hey, why is this dog coming to us? Remove disfluencies in the given sentence.\nOutput: why is this dog coming to us?\nInput: should we we should go tomorrow. Remove disfluencies in the given sentence.\nOutput: we should go tomorrow.\nInput: His, I mean he bought a new car. Remove disfluencies in the given sentence.\nOutput: he bought a new car.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    marathi_translation = f"Input: uhh Tell me, is this not a service? Remove disfluencies in the given sentence.\nOutput: Tell me, is this not a service?\nInput: I will discuss my my work tomorrow. Remove disfluencies in the given sentence.\nOutput: I will discuss my work tomorrow.\nInput: Vajpayee was also among these  econ economists. Remove disfluencies in the given sentence.\nOutput: Vajpayee was also among these economists.\nInput: I thought, um...maybe I'll go see a movie tonight? Remove disfluencies in the given sentence.\nOutput: maybe I'll go see a movie tonight?\nInput: Hey, why is this dog coming to us? Remove disfluencies in the given sentence.\nOutput: why is this dog coming to us?\nInput: should we we should go tomorrow. Remove disfluencies in the given sentence.\nOutput: we should go tomorrow.\nInput: His, I mean he bought a new car. Remove disfluencies in the given sentence.\nOutput: he bought a new car.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    bengali_cs_natural = f"Input: আহ আমাকে বলুন, এটা কি service নয়? Remove disfluencies in the given sentence.\nOutput: আমাকে বলুন, এটা কি service নয়?\nInput: But but এখানে আটকালে তো কাজ হবেনা। Remove disfluencies in the given sentence.\nOutput: But এখানে আটকালে তো কাজ হবেনা।\nInput: বাজপেয়ী জিও এইসব econ economists দের মধ্যে অন্তর্ভুক্ত ছিলেন। Remove disfluencies in the given sentence.\nOutput: বাজপেয়ী জিও এইসব economists দের মধ্যে অন্তর্ভুক্ত ছিলেন।\nInput: আমি ভাবলাম কি যে, আঃ আজকে evening এ কি আমি cinema দেখতে যাব? Remove disfluencies in the given sentence.\nOutput: আজকে evening এ কি আমি cinema দেখতে যাব?\nInput: আরেহ এই dog টা আমাদের দিকে কেন আসছে? Remove disfluencies in the given sentence.\nOutput: এই dog টা আমাদের দিকে কেন আসছে?\nInput: আমিাদের কি কাল, আমাদের কাল যাওয়া উচিত। Remove disfluencies in the given sentence.\nOutput: আমাদের কাল যাওয়া উচিত।\nInput: ওনার, মানে উনি নিজের new car কিনেছেন। Remove disfluencies in the given sentence.\nOutput: উনি নিজের new car কিনেছেন।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    # telugu_prompt2 = f"Input: నేను ఉహుహు... ప్రపంచ కప్ మ్యాచ్ చూడాలనుకుంటున్నాను. Remove disfluencies in the given sentence.\nOutput: నేను ప్రపంచ కప్ మ్యాచ్ చూడాలనుకుంటున్నాను.\nInput: అరేరే నేను నా మొబైల్ ఎక్కడ పెట్టానో మరచిపోయాను! Remove disfluencies in the given sentence.\nOutput: నేను నా మొబైల్ ఎక్కడ పెట్టానో మరచిపోయాను!\nInput: ఆయన బాగా బాగా ఆడుతున్నాడు క్రికెట్. Remove disfluencies in the given sentence.\nOutput: ఆయన బాగా ఆడుతున్నాడు క్రికెట్.\nInput: మన ప్రయాణం శుక్రవా శనివారం నాడు కాదండి? Remove disfluencies in the given sentence.\nOutput: మన ప్రయాణం శనివారం నాడు కాదండి?\nInput: 3:30 కాదు కాదు, 4:30 కి మన ప్రయాణం. Remove disfluencies in the given sentence.\nOutput: 4:30 కి మన ప్రయాణం.\nInput: నా పుస్తకం నాకు పేపర్ కావాలి. Remove disfluencies in the given sentence.\nOutput: నాకు పేపర్ కావాలి.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    telugu_natural = f"Input: నేను ఉహుహు... world cup match చూడాలనుకుంటున్నాను. Remove disfluencies in the given sentence.\nOutput: నేను world cup match చూడాలనుకుంటున్నాను.\nInput: అరేరే నేను నా mobile ఎక్కడ పెట్టానో మరచిపోయాను! Remove disfluencies in the given sentence.\nOutput: నేను నా mobile ఎక్కడ పెట్టానో మరచిపోయాను!\nInput: ఆయన బాగా బాగా ఆడుతున్నాడు cricket. Remove disfluencies in the given sentence.\nOutput: ఆయన బాగా ఆడుతున్నాడు cricket.\nInput: మన ప్రయాణం fri saturady నాడు కాదండి? Remove disfluencies in the given sentence.\nOutput: మన ప్రయాణం saturday నాడు కాదండి?\nInput: 3:30 కాదు కాదు, 4:30 కి మన travel. Remove disfluencies in the given sentence.\nOutput: 4:30 కి మన travel.\nInput: నా book నాకు paper కావాలి. Remove disfluencies in the given sentence.\nOutput: నాకు paper కావాలి.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_natural = f"Input: अ मुझे बताइए ये service नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे बताइए ये service नहीं है क्या?\nInput: But यहां पर पर अटकने से काम नहीं होगा। Remove disfluencies in the given sentence.\nOutput: But यहां पर अटकने से काम नहीं होगा।\nInput: बाजपेयी भी इन अर economists में included थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी भी इन economists में included थे।\nInput: मैंने सोचा कि, उम्म...क्या मैं आज शाम को film देखने जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज शाम को film देखने जाऊँ?\nInput: अरे, यह dog हमारे पास क्यों आ रहा है? Remove disfluencies in the given sentence.\nOutput: यह dog हमारे पास क्यों आ रहा है?\nInput: क्या हमें कल.. हमें कल चलना चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल चलना चाहिए।\nInput: उसके...मतलब, उसने अपनी new car खरीदी है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी new car खरीदी है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    marathi_natural = f"Input: अ मला सांगा, ही service नाही का? Remove disfluencies in the given sentence.\nOutput: मला सांगा, ही service नाही का?\nInput: माझ्या माझ्या कामाची discussion आहे उद्या. Remove disfluencies in the given sentence.\nOutput: माझ्या कामाची discussion आहे उद्या.\nInput: या अर अर्थतज्ज्ञांमध्ये वाजपेयींचाही समावेश होता. Remove disfluencies in the given sentence.\nOutput: या अर्थतज्ज्ञांमध्ये वाजपेयींचाही समावेश होता.\nInput: मी विचार केला कि, अं...मी आज सायंकाळी मी film पहायला जाऊ का? Remove disfluencies in the given sentence.\nOutput: आज सायंकाळी मी film पहायला जाऊ का?\nInput: अरे हा कुत्रा आमच्याजवळ का येतोय? Remove disfluencies in the given sentence.\nOutput: हा कुत्रा आमच्याजवळ का येतोय?\nInput: का आपण उद्या.. आपण उद्या जायला हवं. Remove disfluencies in the given sentence.\nOutput: आपण उद्या जायला हवं.\nInput: त्याची...म्हणजे, त्याने त्याची नवीन car खरेदी केली आहे. Remove disfluencies in the given sentence.\nOutput: त्याने त्याची नवीन car खरेदी केली आहे.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    
    
    telugu_r_30p = f"Input: నేను ఉహుహు... ప్రపంచ కప్ మ్యాచ్ I want to see. Remove disfluencies in the given sentence.\nOutput: నేను ప్రపంచ కప్ మ్యాచ్ I want to see.\nInput: అరేరే నేను నా మొబైల్ ఎక్కడ పెట్టానో I forgot! Remove disfluencies in the given sentence.\nOutput: నేను నా మొబైల్ ఎక్కడ పెట్టానో I forgot! \nInput: ఆయన Very well బాగా playing cricket. Remove disfluencies in the given sentence.\nOutput: ఆయన బాగా playing cricket.\nInput: Our travel fri saturday నాడు కాదండి? Remove disfluencies in the given sentence.\nOutput: Our travel saturday నాడు కాదండి?\nInput: 3:30 కాదు కాదు, 4:30 కి our travel. Remove disfluencies in the given sentence.\nOutput: 4:30 కి our travel.\nInput: నా book I need paper కావాలి. Remove disfluencies in the given sentence.\nOutput: I need paper కావాలి.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    telugu_r_50p = f"Input: I uhh.. ప్రపంచ cup match I want to see. Remove disfluencies in the given sentence.\nOutput: I ప్రపంచ cup match I want to see.\nInput: అరేరే I నా మొబైల్ Where పెట్టానో I forgot! Remove disfluencies in the given sentence.\nOutput: I నా మొబైల్ Where పెట్టానో I forgot!\nInput: ఆయన Very well బాగా playing cricket. Remove disfluencies in the given sentence.\nOutput: ఆయన బాగా playing cricket.\nInput: Our travel fri saturday day కాదండి? Remove disfluencies in the given sentence.\nOutput: Our travel saturday day కాదండి?\nInput: 3:30 కాదు కాదు, 4:30 కి our travel. Remove disfluencies in the given sentence.\nOutput: 4:30 కి our travel.\nInput: నా book I need paper want. Remove disfluencies in the given sentence.\nOutput: I need paper want.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    telugu_r_80p = f"Input: I uhh.. World Cup Match చూడాలనుకుంటున్నాను. Remove disfluencies in the given sentence.\nOutput: I World Cup Match చూడాలనుకుంటున్నాను.\nInput: What a surprise! I My మొబైల్ ఎక్కడ put I forgot! Remove disfluencies in the given sentence.\nOutput: I My మొబైల్ ఎక్కడ put I forgot!\nInput: He very well బాగా ఆడుతున్నాడు cricket. Remove disfluencies in the given sentence.\nOutput: He బాగా ఆడుతున్నాడు cricket.\nInput: Our travel fri saturday day కాదండి? Remove disfluencies in the given sentence.\nOutput: Our travel saturday day కాదండి?\nInput: 3:30 కాదు కాదు, 4:30 at our travel. Remove disfluencies in the given sentence.\nOutput: 4:30 at our travel.\nInput: My book నాకు paper want. Remove disfluencies in the given sentence.\nOutput: నాకు paper want.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    telugu_r_100p = f"Input: I uhh… world cup match I want to see. Remove disfluencies in the given sentence.\nOutput:  I world cup match I want to see.\nInput: What a surprise! I My Mobile phone Where put I forgot! Remove disfluencies in the given sentence.\nOutput: My Mobile phone Where put I forgot!\nInput: He did very well very well playing cricket. Remove disfluencies in the given sentence.\nOutput: He did very well playing cricket.\nInput: Our travel fri saturday day isn’t it? Remove disfluencies in the given sentence.\nOutput: Our travel saturday day isn’t it?\nInput: 3:30 no no, 4:30 at our travel. Remove disfluencies in the given sentence.\nOutput: 4:30 at our travel.\nInput: My book I need paper want. Remove disfluencies in the given sentence.\nOutput: I need paper want.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    telugu_union = f"Input: నేను ఉహుహు... ప్రపంచ కప్ మ్యాచ్ చూడాలనుకుంటున్నాను. Remove disfluencies in the given sentence.\nOutput: నేను ప్రపంచ కప్ మ్యాచ్ చూడాలనుకుంటున్నాను.\nInput: I umm.. want to watch the World Cup match. Remove disfluencies in the given sentence.\nOutput: I want to watch the World Cup match.\nInput: అరేరే నేను నా మొబైల్ ఎక్కడ పెట్టానో మరచిపోయాను! Remove disfluencies in the given sentence.\nOutput: నేను నా మొబైల్ ఎక్కడ పెట్టానో మరచిపోయాను!\nInput: Oh, I forgot where I put my mobile! Remove disfluencies in the given sentence.\nOutput: I forgot where I put my mobile!\nInput: ఆయన బాగా బాగా ఆడుతున్నాడు క్రికెట్. Remove disfluencies in the given sentence.\nOutput: ఆయన బాగా ఆడుతున్నాడు క్రికెట్.\nInput: He is playing playing cricket very well. Remove disfluencies in the given sentence.\nOutput: He is playing cricket very well.\nInput: మన ప్రయాణం శుక్రవా శనివారం నాడు కాదండి? Remove disfluencies in the given sentence.\nOutput: మన ప్రయాణం శనివారం నాడు కాదండి?\nInput: Our trip is on Fri Saturday, isn't it? Remove disfluencies in the given sentence.\nOutput: Our trip is on Saturday, isn't it?\nInput: 3:30 కాదు కాదు, 4:30 కి మన ప్రయాణం. Remove disfluencies in the given sentence.\nOutput: 4:30 కి మన ప్రయాణం.\nInput: Our journey is at 3:30 no no 4:30. Remove disfluencies in the given sentence.\nOutput: Our journey is at 4:30.\nInput: నా పుస్తకం నాకు పేపర్ కావాలి. Remove disfluencies in the given sentence.\nOutput: నాకు పేపర్ కావాలి.\nInput: my book I need a paper. Remove disfluencies in the given sentence.\nOutput: I need a paper.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    telugu_selfTranslate = f"Input: Translate to English, remove disfluencies and Translate to Telugu. నేను ఉహుహు... ప్రపంచ కప్ మ్యాచ్ చూడాలనుకుంటున్నాను.\nOutput: నేను ప్రపంచ కప్ మ్యాచ్ చూడాలనుకుంటున్నాను.\nInput: Translate to English, remove disfluencies and Translate to Telugu. అరేరే నేను నా మొబైల్ ఎక్కడ పెట్టానో మరచిపోయాను!\nOutput: నేను నా మొబైల్ ఎక్కడ పెట్టానో మరచిపోయాను!\nInput: Translate to English, remove disfluencies and Translate to Telugu. ఆయన బాగా బాగా ఆడుతున్నాడు క్రికెట్.\nOutput: ఆయన బాగా ఆడుతున్నాడు క్రికెట్.\nInput: Translate to English, remove disfluencies and Translate to Telugu. మన ప్రయాణం శుక్రవా శనివారం నాడు కాదండి?\nOutput: మన ప్రయాణం శనివారం నాడు కాదండి?\nInput: Translate to English, remove disfluencies and Translate to Telugu. 3:30 కాదు కాదు, 4:30 కి మన ప్రయాణం.\nOutput: 4:30 కి మన ప్రయాణం.\nInput: Translate to English, remove disfluencies and Translate to Telugu. నా పుస్తకం నాకు పేపర్ కావాలి.\nOutput: నాకు పేపర్ కావాలి.\nInput: Translate to English, remove disfluencies and Translate to Telugu. {text_input}\nOutput: "
    hindi_union = f"Input: अ मुझे बताइए ये सेवा नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे बताइए ये सेवा नहीं है क्या?\nInput: uhh Tell me, is this not a service? Remove disfluencies in the given sentence.\nOutput: Tell me, is this not a service?\nInput: लेकिन यहां पर पर अटकने से काम नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन यहां पर अटकने से काम नहीं होगा।\nInput: But but it will not work by staying here. Remove disfluencies in the given sentence.\nOutput:But it will not work by staying here.\nInput: बाजपेयी भी इन अर अर्थशास्त्रियों में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी भी इन अर्थशास्त्रियों में शामिल थे।\nInput: Vajpayee was also among these  econ economists. Remove disfluencies in the given sentence.\nOutput: Vajpayee was also among these economists.\nInput: मैंने सोचा कि, उम्म...क्या मैं आज शाम को फिल्म देखने जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज शाम को फिल्म देखने जाऊँ?\nInput: I thought, um...maybe I'll go see a movie tonight? Remove disfluencies in the given sentence.\nOutput: maybe I'll go see a movie tonight?\nInput: अरे, यह कुत्ता हमारे पास क्यों आ रहा है? Remove disfluencies in the given sentence.\nOutput: यह कुत्ता हमारे पास क्यों आ रहा है?\nInput: Hey, why is this dog coming to us? Remove disfluencies in the given sentence.\nOutput: why is this dog coming to us?\nInput: क्या हमें कल.. हमें कल चलना चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल चलना चाहिए।\nInput: should we we should go tomorrow. Remove disfluencies in the given sentence.\nOutput: we should go tomorrow.\nInput: उसके...मतलब, उसने अपनी नई कार खरीदी है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी नई कार खरीदी है।\nInput: His, I mean he bought a new car. Remove disfluencies in the given sentence.\nOutput: he bought a new car.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_selfTranslate = f"Input: Translate to English, remove disfluencies and Translate to Hindi. अ मुझे बताइए ये सेवा नहीं है क्या?\nOutput: मुझे बताइए ये सेवा नहीं है क्या?\nInput: Translate to English, remove disfluencies and Translate to Hindi. लेकिन यहां पर पर अटकने से काम नहीं होगा।\nOutput: लेकिन यहां पर अटकने से काम नहीं होगा।\nInput: Translate to English, remove disfluencies and Translate to Hindi. बाजपेयी भी इन अर अर्थशास्त्रियों में शामिल थे।\nOutput: बाजपेयी भी इन अर्थशास्त्रियों में शामिल थे।\nInput: Translate to English, remove disfluencies and Translate to Hindi. मैंने सोचा कि, उम्म...क्या मैं आज शाम को फिल्म देखने जाऊँ?\nOutput: क्या मैं आज शाम को फिल्म देखने जाऊँ?\nInput: Translate to English, remove disfluencies and Translate to Hindi. अरे, यह कुत्ता हमारे पास क्यों आ रहा है?\nOutput: यह कुत्ता हमारे पास क्यों आ रहा है?\nInput: Translate to English, remove disfluencies and Translate to Hindi. क्या हमें कल.. हमें कल चलना चाहिए।\nOutput: हमें कल चलना चाहिए।\nInput: Translate to English, remove disfluencies and Translate to Hindi. उसके...मतलब, उसने अपनी नई कार खरीदी है।\nOutput: उसने अपनी नई कार खरीदी है।\nInput: Translate to English, remove disfluencies and Translate to Hindi. {text_input}\nOutput: "

    hindi_30p_align = f"Input: अ मुझे tell ये सेवा नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे tell ये सेवा नहीं है क्या?\nInput: लेकिन यहां पर पर अटकने से work नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन यहां पर अटकने से work नहीं होगा।\nInput: Vajpayee also इन अर अर्थशास्त्रियों में शामिल थे। Remove disfluencies in the given sentence.\nOutput: Vajpayee also इन अर्थशास्त्रियों में शामिल थे।\nInput: मैंने सोचा कि, उम्म...क्या मैं आज शाम को फिल्म see जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज शाम को फिल्म see जाऊँ?\nInput: अरे, यह कुत्ता हमारे near क्यों आ रहा है? Remove disfluencies in the given sentence.\nOutput: यह कुत्ता हमारे near क्यों आ रहा है?\nInput: क्या हमें कल.. हमें कल walk चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल walk चाहिए।\nInput: उसके...मतलब, उसने अपनी नई car खरीदी है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी नई car खरीदी है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_50p_align = f"Input: अ मुझे tell ये सेवा not is क्या? Remove disfluencies in the given sentence.\nOutput: मुझे tell ये सेवा not is क्या?\nInput: लेकिन यहां पर पर stuck से work not होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन यहां पर stuck से work not होगा।\nInput: Vajpayee also इन अर अर्थशास्त्रियों में शामिल थे। Remove disfluencies in the given sentence.\nOutput: Vajpayee also इन अर्थशास्त्रियों में शामिल थे।\nInput: मैंने सोचा कि, उम्म...क्या मैं आज शाम को moview see जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज शाम को movie see जाऊँ\nInput: अरे, यह dog हमारे near क्यों आ रहा है? Remove disfluencies in the given sentence.\nOutput: यह dog हमारे near क्यों आ रहा है?\nInput: क्या हमें कल.. हमें कल go चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल go चाहिए।\nInput: उसके...मतलब, उसने अपनी new car bought है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी new car bought है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_80p_align = f"Input: अ मुझे tell ये service not is क्या? Remove disfluencies in the given sentence.\nOutput: मुझे tell ये service not is क्या?\nInput: लेकिन यहां पर पर stuck से work not होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन यहां पर stuck से work not होगा।\nInput: Vajpayee also इन अर economists में शामिल was। Remove disfluencies in the given sentence.\nOutput: Vajpayee also इन economists में शामिल was।\nInput: मैंने thought कि, उम्म...क्या मैं आज evening को film see जाऊँ? Remove disfluencies in the given sentence.\nOutput: क्या मैं आज evening को film see जाऊँ?\nInput: अरे, यह dog हमारे near क्यों coming रहा है? Remove disfluencies in the given sentence.\nOutput: यह dog हमारे near क्यों coming रहा है?\nInput: क्या हमें tomorrow.. हमें कल walk चाहिए। Remove disfluencies in the given sentence.\nOutput: हमें कल walk चाहिए।\nInput: उसके...मतलब, उसने अपनी new car bought है। Remove disfluencies in the given sentence.\nOutput: उसने अपनी new car bought है।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    
    

    prompt_reorganized = (
    "Input: I umm.. don't want to go uhh.. to the school. Remove disfluencies in the given sentence.\nOutput: I don't want to go to the school.\n"
    "Input: The sun sun rises in the east. Remove disfluencies in the given sentence.\nOutput: The sun rises in the east.\n"
    "Input: There are nin eight planets in the solar system. Remove disfluencies in the given sentence.\nOutput: There are eight planets in the solar system.\n"
    "Input: Let's meet at 2 sorry 3. Remove disfluencies in the given sentence.\nOutput: Let's meet at 3.\n"
    "Input: Ouch! It hurts. Remove disfluencies in the given sentence.\nOutput: It hurts.\n"
    "Input: Well, I don't think I should try this. Remove disfluencies in the given sentence.\nOutput: I don't think I should try this.\n"
    "Input: Saketh and I drove I'm hungry now is there something to eat? Remove disfluencies in the given sentence.\nOutput: I'm hungry now is there something to eat?\n"
    "Input: I went to the market today. Remove disfluencies in the given sentence.\nOutput: I went to the market today.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_de = (
    "Input: Bitte schreibe eine E-Mail an, an max@moritz.de Remove disfluencies in the given sentence.\nOutput: Bitte schreibe eine E-Mail an max@moritz.de.\n"
    "Input: Sende eine Videobotschaft per öhm per Skype an 7867857567. Remove disfluencies in the given sentence.\nOutput: Sende eine Videobotschaft per Skype an 7867857567.\n"
    "Input: Schreibe eine Mail an ja eine Mail an Peter Müller. Remove disfluencies in the given sentence.\nOutput: Schreibe eine Mail an Peter Müller.\n"
    "Input: Kannst du eine SMS an Gina senden und uhh ja füge bitte meinen Standort an. Remove disfluencies in the given sentence.\nOutput: Kannst du eine SMS an Gina senden und füge bitte meinen Standort an.\n"
    "Input: Schicke, äh ja, schick das an Ann. Remove disfluencies in the given sentence.\nOutput: schick das an Ann.\n"
    "Input: Kannst du bitte die Nummer ähh von Maria per SMS an Bob schicken. Remove disfluencies in the given sentence.\nOutput: Kannst du bitte die Nummer von Maria per SMS an Bob schicken.\n"
    "Input: Sende eine Nachricht eine Nachricht an Lufthansa. Remove disfluencies in the given sentence.\nOutput: Sende eine Nachricht an Lufthansa.\n"
    "Input: Schreibe eine Whatsapp an 12345 und füge meinen Standort hinzu. Remove disfluencies in the given sentence.\nOutput: Schreibe eine Whatsapp an 12345 und füge meinen Standort hinzu.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_de_30p = (
    "Input: Bitte schreibe eine E-Mail an, an max@moritz.de Remove disfluencies in the given sentence.\nOutput: Bitte schreibe eine E-Mail an max@moritz.de.\n"
    "Input: Send eine Videobotschaft per öhm per Skype an 7867857567. Remove disfluencies in the given sentence.\nOutput: Send eine Videobotschaft per Skype an 7867857567.\n"
    "Input: Schreibe eine Mail an ja eine Mail an Peter Müller. Remove disfluencies in the given sentence.\nOutput: Schreibe eine Mail an Peter Müller.\n"
    "Input: Kannst du eine SMS an Gina send und uhh ja füge bitte meinen Standort an. Remove disfluencies in the given sentence.\nOutput: Kannst du eine SMS an Gina send und füge bitte meinen Standort an.\n"
    "Input: Schicke, äh ja, send das an Ann. Remove disfluencies in the given sentence.\nOutput: send das an Ann.\n"
    "Input: Kannst du please die Nummer ähh von Maria per SMS an Bob schicken. Remove disfluencies in the given sentence.\nOutput: Kannst du please die Nummer von Maria per SMS an Bob schicken.\n"
    "Input: Send eine Nachricht eine Nachricht an Lufthansa. Remove disfluencies in the given sentence.\nOutput: Send eine Nachricht an Lufthansa.\n"
    "Input: Schreibe eine Whatsapp an 12345 und füge meinen Standort hinzu. Remove disfluencies in the given sentence.\nOutput: Schreibe eine Whatsapp an 12345 und füge meinen Standort hinzu.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_de_50p = (
    "Input: Bitte writing eine E-Mail an, an max@moritz.de Remove disfluencies in the given sentence.\nOutput: Bitte writing eine E-Mail an max@moritz.de.\n"
    "Input: Send eine Video message per öhm per Skype an 7867857567. Remove disfluencies in the given sentence.\nOutput: Send eine Video message per Skype an 7867857567.\n"
    "Input: Write eine Mail an ja eine Mail an Peter Müller. Remove disfluencies in the given sentence.\nOutput: Write eine Mail an Peter Müller.\n"
    "Input: Kannst du eine SMS an Gina send und uhh ja füge please meinen Standort an. Remove disfluencies in the given sentence.\nOutput: Kannst du eine SMS an Gina send und füge please meinen Standort an.\n"
    "Input: Schicke, äh ja, send das an Ann. Remove disfluencies in the given sentence.\nOutput: send das an Ann.\n"
    "Input: Can du please die Nummer ähh von Maria per SMS an Bob schicken. Remove disfluencies in the given sentence.\nOutput: Can du please die Nummer von Maria per SMS an Bob schicken.\n"
    "Input: Send eine Nachricht eine message an Lufthansa. Remove disfluencies in the given sentence.\nOutput: Send eine message an Lufthansa.\n"
    "Input: Write eine Whatsapp an 12345 und füge meinen Standort hinzu. Remove disfluencies in the given sentence.\nOutput: Write eine Whatsapp an 12345 und füge meinen Standort hinzu.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_de_80p = (
    "Input: Bitte writing eine E-Mail an, an max@moritz.de Remove disfluencies in the given sentence.\nOutput: Bitte writing eine E-Mail an max@moritz.de.\n"
    "Input: Send eine Video message per öhm per Skype an 7867857567. Remove disfluencies in the given sentence.\nOutput: Send eine Video message per Skype an 7867857567.\n"
    "Input: Write eine Mail an ja eine Mail an Peter Müller. Remove disfluencies in the given sentence.\nOutput: Write eine Mail an Peter Müller.\n"
    "Input: Can du eine SMS an Gina send und uhh ja füge please meinen Standort an. Remove disfluencies in the given sentence.\nOutput: Can du eine SMS an Gina send und füge please meinen Standort an.\n"
    "Input: Schicke, äh ja, schick das an Ann. Remove disfluencies in the given sentence.\nOutput: schick das an Ann.\n"
    "Input: Can du please die number ähh von Maria per SMS an Bob schicken. Remove disfluencies in the given sentence.\nOutput: Can du please die number von Maria per SMS an Bob schicken.\n"
    "Input: Sende eine message eine message an Lufthansa. Remove disfluencies in the given sentence.\nOutput: Sende eine message an Lufthansa.\n"
    "Input: Write eine Whatsapp an 12345 und füge meinen Standort hinzu. Remove disfluencies in the given sentence.\nOutput: Write eine Whatsapp an 12345 und füge meinen Standort hinzu.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_fr = (
    "Input: Je veux que tu envoies la photo photo d' écran à Mireille avec Lucie en cc. Remove disfluencies in the given sentence.\nOutput: Je veux que tu envoies la photo d' écran à Mireille avec Lucie en cc.\n"
    "Input: Envoie un mail à euh jena@polonium.com. Remove disfluencies in the given sentence.\nOutput: Envoie un mail à jena@polonium.com.\n"
    "Input: envoie une un message à Alice. Remove disfluencies in the given sentence.\nOutput: envoie un message à Alice.\n"
    "Input: Peux-tu euuh envoyer cet SMS sur le chien de maman? Remove disfluencies in the given sentence.\nOutput: Peux-tu envoyer cet SMS sur le chien de maman?\n"
    "Input: Dis à Karen par euh SMS que j'arrive en en joignant mon heure d'arrivée. Remove disfluencies in the given sentence.\nOutput: Dis à Karen par SMS que j'arrive en joignant mon heure d'arrivée.\n"
    "Input: Mets en pièce jointe mes coordonnées GPS au courriel pour euh Lucie. Remove disfluencies in the given sentence.\nOutput: Mets en pièce jointe mes coordonnées GPS au courriel pour Lucie.\n"
    "Input: écris une un mail à Alice. Remove disfluencies in the given sentence.\nOutput: écris un mail à Alice.\n"
    "Input: Merci d'écrire un email à Pierre. Remove disfluencies in the given sentence.\nOutput: Merci d'écrire un email à Pierre.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_fr_30p = (
    "Input: Je want que tu envoies la photo photo d' écran à Mireille avec Lucie en cc. Remove disfluencies in the given sentence.\nOutput: Je want que tu envoies la photo d' écran à Mireille avec Lucie en cc.\n"
    "Input: Envoie un mail à euh jena@polonium.com. Remove disfluencies in the given sentence.\nOutput: Envoie un mail à jena@polonium.com.\n"
    "Input: envoie une un message à Alice. Remove disfluencies in the given sentence.\nOutput: envoie un message à Alice.\n"
    "Input: Peux-tu euuh envoyer cet SMS sur le dogs de maman? Remove disfluencies in the given sentence.\nOutput: Peux-tu envoyer cet SMS sur le dogs de maman?\n"
    "Input: Dis à Karen par euh SMS que j'arrive en en joignant mon heure d'arrivée. Remove disfluencies in the given sentence.\nOutput: Dis à Karen par SMS que j'arrive en joignant mon heure d'arrivée.\n"
    "Input: Mets en pièce jointe mes coordinates GPS au courriel pour euh Lucie. Remove disfluencies in the given sentence.\nOutput: Mets en pièce jointe mes coordinates GPS au courriel pour Lucie.\n"
    "Input: écris une un mail à Alice. Remove disfluencies in the given sentence.\nOutput: écris un mail à Alice.\n"
    "Input: Merci d'écrire un email à Pierre. Remove disfluencies in the given sentence.\nOutput: Merci d'écrire un email à Pierre.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_fr_50p = (
    "Input: Je want que tu envoies la picture picture d' écran à Mireille avec Lucie en cc. Remove disfluencies in the given sentence.\nOutput: Je want que tu envoies la picture d' écran à Mireille avec Lucie en cc.\n"
    "Input: Send un mail à euh jena@polonium.com. Remove disfluencies in the given sentence.\nOutput: Send un mail à jena@polonium.com.\n"
    "Input: send une un message à Alice. Remove disfluencies in the given sentence.\nOutput: send un message à Alice.\n"
    "Input: Peux-tu euuh send cet SMS sur le dogs de maman? Remove disfluencies in the given sentence.\nOutput: Peux-tu send cet SMS sur le dogs de maman?\n"
    "Input: Dis à Karen par euh SMS que j'arrive en en joignant mon heure d'arrivée. Remove disfluencies in the given sentence.\nOutput: Dis à Karen par SMS que j'arrive en joignant mon heure d'arrivée.\n"
    "Input: Mets en pièce jointe mes coordinates GPS au email pour euh Lucie. Remove disfluencies in the given sentence.\nOutput: Mets en pièce jointe mes coordinates GPS au email pour Lucie.\n"
    "Input: écris une un mail à Alice. Remove disfluencies in the given sentence.\nOutput: écris un mail à Alice.\n"
    "Input: Merci d'writing un email à Pierre. Remove disfluencies in the given sentence.\nOutput: Merci d'writing un email à Pierre.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_fr_80p = (
    "Input: Je veux que tu envoies la picture picture d' screen à Mireille avec Lucie en cc. Remove disfluencies in the given sentence.\nOutput: Je veux que tu envoies la picture d' screen à Mireille avec Lucie en cc.\n"
    "Input: Send un mail à euh jena@polonium.com. Remove disfluencies in the given sentence.\nOutput: Send un mail à jena@polonium.com.\n"
    "Input: send une un message à Alice. Remove disfluencies in the given sentence.\nOutput: send un message à Alice.\n"
    "Input: Can-tu euuh send cet SMS sur le dogs de maman? Remove disfluencies in the given sentence.\nOutput: Can-tu send cet SMS sur le dogs de maman?\n"
    "Input: Dis à Karen par euh SMS que j'arrives en en joignant mon heure d'arrivée. Remove disfluencies in the given sentence.\nOutput: Dis à Karen par SMS que j'arrives en joignant mon heure d'arrivée.\n"
    "Input: Mets en coin jointe mes coordinates GPS au email pour euh Lucie. Remove disfluencies in the given sentence.\nOutput: Mets en coin jointe mes coordinates GPS au email pour Lucie.\n"
    "Input: écris une un mail à Alice. Remove disfluencies in the given sentence.\nOutput: écris un mail à Alice.\n"
    "Input: Thanks d'writind un email à Pierre. Remove disfluencies in the given sentence.\nOutput: Thanks d'writind un email à Pierre.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_fr_30p_align = (
    "Input: Je veux que tu envoies la screenshot screenshot d' écran à Mireille avec Lucie en cc. Remove disfluencies in the given sentence.\nOutput: Je veux que tu envoies la screenshot d' écran à Mireille avec Lucie en cc.\n"
    "Input: Send un mail à euh jena@polonium.com. Remove disfluencies in the given sentence.\nOutput: Send un mail à jena@polonium.com.\n"
    "Input: envoie une un message à Alice. Remove disfluencies in the given sentence.\nOutput: envoie un message à Alice.\n"
    "Input: Peux-tu euuh envoyer cet SMS sur le chien de maman? Remove disfluencies in the given sentence.\nOutput: Peux-tu envoyer cet SMS sur le chien de maman?\n"
    "Input: Dis à Karen par euh SMS que j'arrive en en joignant mon heure d'arrivée. Remove disfluencies in the given sentence.\nOutput: Dis à Karen par SMS que j'arrive en joignant mon heure d'arrivée.\n"
    "Input: Mets en pièce jointe mes coordinates GPS au courriel pour euh Lucie. Remove disfluencies in the given sentence.\nOutput: Mets en pièce jointe mes coordinates GPS au courriel pour Lucie.\n"
    "Input: écris une un mail à Alice. Remove disfluencies in the given sentence.\nOutput: écris un mail à Alice.\n"
    "Input: Thank writing un email à Pierre. Remove disfluencies in the given sentence.\nOutput: Thank writing un email à Pierre.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_fr_50p_align = (
    "Input: Je veux que tu send la screenshot screenshot d' écran à Mireille avec Lucie en cc. Remove disfluencies in the given sentence.\nOutput: Je veux que tu send la screenshot d' écran à Mireille avec Lucie en cc.\n"
    "Input: Send un mail à euh jena@polonium.com. Remove disfluencies in the given sentence.\nOutput: Send un mail à jena@polonium.com.\n"
    "Input: send une un message à Alice. Remove disfluencies in the given sentence.\nOutput: send un message à Alice.\n"
    "Input: Peux-tu euuh envoyer cet SMS sur le chien de maman? Remove disfluencies in the given sentence.\nOutput: Peux-tu envoyer cet SMS sur le chien de maman?\n"
    "Input: Dis à Karen par euh SMS que j'arrive en en joignant mon heure d'arrivée. Remove disfluencies in the given sentence.\nOutput: Dis à Karen par SMS que j'arrive en joignant mon heure d'arrivée.\n"
    "Input: Mets en pièce jointe mes coordinates GPS au courriel pour euh Lucie. Remove disfluencies in the given sentence.\nOutput: Mets en pièce jointe mes coordinates GPS au courriel pour Lucie.\n"
    "Input: Write une un mail à Alice. Remove disfluencies in the given sentence.\nOutput: Write un mail à Alice.\n"
    "Input: Thank writing un email à Pierre. Remove disfluencies in the given sentence.\nOutput: Thank writing un email à Pierre.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_fr_80p_align = (
    "Input: Je want que tu send la screenshot screenshot d' écran à Mireille avec Lucie en cc. Remove disfluencies in the given sentence.\nOutput: Je want que tu send la screenshot d' écran à Mireille avec Lucie en cc.\n"
    "Input: Send un mail à euh jena@polonium.com. Remove disfluencies in the given sentence.\nOutput: Send un mail à jena@polonium.com.\n"
    "Input: send une un message à Alice. Remove disfluencies in the given sentence.\nOutput: send un message à Alice.\n"
    "Input: Peux-tu euuh send cet SMS sur le chien de maman? Remove disfluencies in the given sentence.\nOutput: Peux-tu send cet SMS sur le chien de maman?\n"
    "Input: Dis à Karen par euh SMS que j'arrive en en joignant mon heure d'arrivée. Remove disfluencies in the given sentence.\nOutput: Dis à Karen par SMS que j'arrive en joignant mon heure d'arrivée.\n"
    "Input: Mets en pièce jointe mes coordinates GPS au email pour euh Lucie. Remove disfluencies in the given sentence.\nOutput: Mets en pièce jointe mes coordinates GPS au email pour Lucie.\n"
    "Input: Write une un mail à Alice. Remove disfluencies in the given sentence.\nOutput: Write un mail à Alice.\n"
    "Input: Thank writing un email à Pierre. Remove disfluencies in the given sentence.\nOutput: Thank writing un email à Pierre.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_vi = (
    "Input: tôi cần thuê à tôi muốn bay một chuyến khứ hồi từ đà nẵng đến đà lạt. Remove disfluencies in the given sentence.\nOutput: tôi muốn bay một chuyến khứ hồi từ đà nẵng đến đà lạt.\n"
    "Input: sân bay ừm không hãng hàng không nào có đường bay từ bắc kinh ờ ý tôi là thượng thượng hải đến washington dc mà cần nối chuyến qua các thành phố khác. Remove disfluencies in the given sentence.\nOutput: hãng hàng không nào có đường bay từ thượng hải đến washington dc mà cần nối chuyến qua các thành phố khác.\n"
    "Input: cho tôi biết tất cả các máy bay à chuyến bay từ huế đến quy nhơn. Remove disfluencies in the given sentence.\nOutput: cho tôi biết tất cả các chuyến bay từ huế đến quy nhơn.\n"
    "Input: đà nẵng đến ờ hồ chí minh í lộn đến cà mau. Remove disfluencies in the given sentence.\nOutput: đà nẵng đến cà mau.\n"
    "Input: có bao nhiêu ghế à ý tôi là hạng ghế. Remove disfluencies in the given sentence.\nOutput: có bao nhiêu hạng ghế.\n"
    "Input: chuyến bay nào rời buôn ma thuột vào ngày mùng 4 ờ không ngày 5 tháng 7 sau 7 giờ tối và đến cần thơ. Remove disfluencies in the given sentence.\nOutput: chuyến bay nào rời buôn ma thuột vào ngày 5 tháng 7 sau 7 giờ tối và đến cần thơ.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_vi_30p = (
    "Input: tôi cần rented à tôi muốn fly một chuyến khứ hồi từ đà nẵng đến đà lạt. Remove disfluencies in the given sentence.\nOutput: tôi muốn fly một chuyến khứ hồi từ đà nẵng đến đà lạt.\n"
    "Input: airport ừm không company aviation nào có đường bay từ bắc kinh ờ ý tôi là thượng thượng hải đến washington dc mà cần nối chuyến qua các thành phố khác. Remove disfluencies in the given sentence.\nOutput: company aviation nào có đường bay từ thượng hải đến washington dc mà cần nối chuyến qua các thành phố khác.\n"
    "Input: cho tôi know tất cả các máy bay à chuyến bay từ huế đến quy nhơn. Remove disfluencies in the given sentence.\nOutput: cho tôi know tất cả các chuyến bay từ huế đến quy nhơn.\n"
    "Input: Da Nang đến ờ hồ chí minh í lộn đến cà mau. Remove disfluencies in the given sentence.\nOutput: Da Nang đến cà mau.\n"
    "Input: có how many ghế à ý I là hạng ghế. Remove disfluencies in the given sentence.\nOutput: có how many hạng ghế.\n"
    "Input: flight which depart buôn ma thuột vào ngày mùng 4 ờ không ngày 5 tháng 7 sau 7 giờ tối và đến cần thơ. Remove disfluencies in the given sentence.\nOutput: flight which depart buôn ma thuột vào ngày 5 tháng 7 sau 7 giờ tối và đến cần thơ.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_vi_50p = (
    "Input: tôi need rented à tôi muốn fly một chuyến khứ hồi từ đà nẵng đến đà lạt. Remove disfluencies in the given sentence.\nOutput: tôi muốn fly một chuyến khứ hồi từ đà nẵng đến đà lạt.\n"
    "Input: airport ừm không company aviation nào có flight route từ Beijing ờ ý tôi là thượng thượng hải đến washington dc mà cần nối chuyến qua các thành phố khác. Remove disfluencies in the given sentence.\nOutput: company aviation nào có flight route từ thượng hải đến washington dc mà cần nối chuyến qua các thành phố khác.\n"
    "Input: cho tôi know tất cả các máy bay à flight từ huế đến quy nhơn. Remove disfluencies in the given sentence.\nOutput: cho tôi know tất cả các flight từ huế đến quy nhơn.\n"
    "Input: Da Nang đến ờ Ho Chi Minh í lộn đến cà mau. Remove disfluencies in the given sentence.\nOutput: Da Nang đến cà mau.\n"
    "Input: có how many ghế à meaning tôi là hạng ghế. Remove disfluencies in the given sentence.\nOutput: có how many hạng ghế.\n"
    "Input: flight which depart buôn ma thuột vào ngày mùng 4 ờ không 5th day July sau 7 giờ tối và đến cần thơ. Remove disfluencies in the given sentence.\nOutput: flight which depart buôn ma thuột vào 5th day July sau 7 giờ tối và đến cần thơ.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_vi_80p= (
    "Input: tôi need rented à tôi muốn fly một chuyến khứ hồi từ đà nẵng đến đà wind. Remove disfluencies in the given sentence.\nOutput: tôi muốn fly một chuyến khứ hồi từ đà nẵng đến đà wind.\n"
    "Input: airport ừm không company aviation nào có flight route từ Beijing ờ intention tôi là Shanghai Shanghai đến washington dc mà cần connecting flight qua các thành phố khác. Remove disfluencies in the given sentence.\nOutput: company aviation nào có flight route từ Shanghai đến washington dc mà cần connecting flight qua các thành phố khác.\n"
    "Input: cho tôi know tất cả các máy bay à flight từ Hue đến quy nhơn. Remove disfluencies in the given sentence.\nOutput: cho tôi know tất cả các flight từ Hue đến quy nhơn.\n"
    "Input: Da Nang đến ờ Ho Chi Minh í mistakenly đến cà mau. Remove disfluencies in the given sentence.\nOutput: Da Nang đến cà mau.\n"
    "Input: có how many ghế à meaning I là hạng ghế. Remove disfluencies in the given sentence.\nOutput: có how many hạng ghế.\n"
    "Input: flight which depart buôn ma thuột vào ngày mùng 4 ờ không 5th day July after 7 o'clock in the evening và arrive cần thơ. Remove disfluencies in the given sentence.\nOutput: flight which depart buôn ma thuột vào 5th day July after 7 o'clock in the evening và arrive cần thơ.\n"
    f"Input: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    )

    prompt_tr = (
    "Input: ismail abimiz şımarmaz şımarırsa da şımarsın zaten hak ediyoo Correct grammatical or spelling errors in the given sentence.\nOutput: İsmail abimiz şımarmaz şımarırsa da şımarsın zaten hak ediyor\n"
    "Input: iyi aksamlar panpisleeer Correct grammatical or spelling errors in the given sentence.\nOutput: iyi akşamlar panpişler\n"
    "Input: benım arkadasım dıye benım halletmem gerekmıyo Correct grammatical or spelling errors in the given sentence.\nOutput: benim arkadaşım diye benim halletmem gerekmiyor\n"
    "Input: gıt mesaj atana sor Correct grammatical or spelling errors in the given sentence.\nOutput: git mesaj atana sor\n"
    "Input: sayenızde dıger arkadasla kötü oldum Correct grammatical or spelling errors in the given sentence.\nOutput: sayenizde diğer arkadaşla kötü oldum\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_tr_30p = (
    "Input: ismail abimiz şımarmaz şımarırsa da şımarsın zaten rights ediyoo Correct grammatical or spelling errors in the given sentence.\nOutput: İsmail abimiz şımarmaz şımarırsa da şımarsın zaten rights ediyor\n"
    "Input: iyi aksamlar panpisleeer Correct grammatical or spelling errors in the given sentence.\nOutput: iyi akşamlar panpişler\n"
    "Input: benım arkadasım dıye benım halletmem gerekmıyo Correct grammatical or spelling errors in the given sentence.\nOutput: benim arkadaşım diye benim halletmem gerekmiyor\n"
    "Input: gıt mesaj atana ask Correct grammatical or spelling errors in the given sentence.\nOutput: git mesaj atana ask\n"
    "Input: sayenızde dıger arkadasla kötü oldum Correct grammatical or spelling errors in the given sentence.\nOutput: sayenizde diğer arkadaşla kötü oldum\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_tr_50p = (
    "Input: ismail abimiz şımarmaz şımarırsa da şımarsın already rights ediyoo Correct grammatical or spelling errors in the given sentence.\nOutput: İsmail abimiz şımarmaz şımarırsa da şımarsın already rights ediyor\n"
    "Input: fine aksamlar panpisleeer Correct grammatical or spelling errors in the given sentence.\nOutput: fine akşamlar panpişler\n"
    "Input: benım arkadasım dıye benım halletmem gerekmıyo Correct grammatical or spelling errors in the given sentence.\nOutput: benim arkadaşım diye benim halletmem gerekmiyor\n"
    "Input: gıt mesaj atana ask Correct grammatical or spelling errors in the given sentence.\nOutput: git mesaj atana ask\n"
    "Input: sayenızde dıger arkadasla wicked oldum Correct grammatical or spelling errors in the given sentence.\nOutput: sayenizde diğer arkadaşla wicked oldum\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_tr_80p = (
    "Input: ismail abimiz şımarmaz şımarırsa da şımarsın already rights ediyoo Correct grammatical or spelling errors in the given sentence.\nOutput: İsmail abimiz şımarmaz şımarırsa da şımarsın already rights ediyor\n"
    "Input: fine aksamlar panpisleeer Correct grammatical or spelling errors in the given sentence.\nOutput: fine akşamlar panpişler\n"
    "Input: benım arkadasım dıye benım halletmem gerekmıyo Correct grammatical or spelling errors in the given sentence.\nOutput: benim arkadaşım diye benim halletmem gerekmiyor\n"
    "Input: gıt messages atana ask Correct grammatical or spelling errors in the given sentence.\nOutput: git messages atana ask\n"
    "Input: sayenızde dıger arkadasla wicked oldum Correct grammatical or spelling errors in the given sentence.\nOutput: sayenizde diğer arkadaşla wicked oldum\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_eng = (
    "Input: She see Tom is catched by policeman in park at last night. Correct grammatical or spelling errors in the given sentence.\nOutput: She saw Tom caught by a policeman in the park last night.\n"
    "Input: I went to the unversity. Correct grammatical or spelling errors in the given sentence.\nOutput: I went to the university.\n"
    "Input: He went to Yeman to study. Correct grammatical or spelling errors in the given sentence.\nOutput: He went to Yemen to study.\n"
    "Input: There is mutlipel train from Bombay to Chennai. Correct grammatical or spelling errors in the given sentence.\nOutput: There are multiple trains from Bombay to Chennai.\n"
    "Input: Do your house has air - conditioning?. Correct grammatical or spelling errors in the given sentence.\nOutput: Does your house have air - conditioning?\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_de_gec = (
    "Input: Und dazu kann ich nur antworten Correct grammatical or spelling errors in the given sentence.\nOutput: Und darauf kann ich nur antworten\n"
    "Input: Hallo Jens , Diene Glückwünsche schon fertig . Correct grammatical or spelling errors in the given sentence.\nOutput: Hallo Jens , deine Glückwünsche sind schon fertig .\n"
    "Input: Du hast shon ein Fater . Correct grammatical or spelling errors in the given sentence.\nOutput: Du hast schon einen Vater .\n"
    "Input: Ich gratulieren . Correct grammatical or spelling errors in the given sentence.\nOutput: Ich gratuliere .\n"
    "Input: Wer hat geboren der Junge oder das Mätchan Wie Name deine Kind ? Correct grammatical or spelling errors in the given sentence.\nOutput: Wer hat geboren , der Junge oder das Mädchen ? Wie ist der Name deines Kindes ?\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_de_gec_30p = (
    "Input: Und dazu can ich nur antworten Correct grammatical or spelling errors in the given sentence.\nOutput: Und darauf can ich nur antworten\n"
    "Input: Hallo Jens , Diene Glückwünsche schon done . Correct grammatical or spelling errors in the given sentence.\nOutput: Hallo Jens , deine Glückwünsche sind schon done .\n"
    "Input: Du hast shon ein Fater . Correct grammatical or spelling errors in the given sentence.\nOutput: Du hast schon einen Vater .\n"
    "Input: Ich gratulieren . Correct grammatical or spelling errors in the given sentence.\nOutput: Ich gratuliere .\n"
    "Input: Wer hat geboren der Junge oder das Mätchan Wie Name deine child ? Correct grammatical or spelling errors in the given sentence.\nOutput: Wer hat geboren , der Junge oder das Mädchen ? Wie ist der Name deines Kindes ?\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_de_gec_50p = (
    "Input: Und dazu can ich nur antworten Correct grammatical or spelling errors in the given sentence.\nOutput: Und darauf can ich nur antworten\n"
    "Input: Hallo Jens , Diene Glückwünsche schon done . Correct grammatical or spelling errors in the given sentence.\nOutput: Hallo Jens , deine Glückwünsche sind schon done .\n"
    "Input: Du hast shon ein Fater . Correct grammatical or spelling errors in the given sentence.\nOutput: Du hast schon einen Vater .\n"
    "Input: Ich gratulieren . Correct grammatical or spelling errors in the given sentence.\nOutput: Ich gratuliere .\n"
    "Input: Wer hat geboren der Junge oder das Mätchan Wie Name deine child ? Correct grammatical or spelling errors in the given sentence.\nOutput: Wer hat geboren , der Junge oder das Mädchen ? Wie ist der Name deines Kindes ?\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_de_gec_80p = (
    "Input: Und dazu can ich nur replies Correct grammatical or spelling errors in the given sentence.\nOutput: Und darauf can ich nur replies\n"
    "Input: Hallo Jens , Diene Glückwünsche already done . Correct grammatical or spelling errors in the given sentence.\nOutput: Hallo Jens , deine Glückwünsche sind already done .\n"
    "Input: Du hast shon ein Fater . Correct grammatical or spelling errors in the given sentence.\nOutput: Du hast schon einen Vater .\n"
    "Input: Ich congrats . Correct grammatical or spelling errors in the given sentence.\nOutput: Ich gratuliere .\n"
    "Input: Wer hat geboren der kid oder das Mätchan Wie Name deine child ? Correct grammatical or spelling errors in the given sentence.\nOutput: Wer hat geboren , der kid oder das Mädchen ? Wie ist der Name deines Kindes ?\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_gr = (
    "Input: Έκλειναν το κεφάλι σε τενεκέ και χτυπούσαν με πλήκτρα να τρελάνουν τον ανακρινόμενο Correct grammatical or spelling errors in the given sentence.\nOutput: Μαλακιζονταν το κεφάλι σε τενεκέ και χτυπούσαν με πλήκτρα να τρελάνουν τον ανακρινόμενο\n"
    "Input: Από το 1996 ο Κούλογλου επιμελούταν και παρουσίαζε την εκπομπή ''Ρεπορτάζ χωρίς Σύνορα'' στη δημόσια τηλεόραση Correct grammatical or spelling errors in the given sentence.\nOutput: Από το 1996 ο Κούλογλου είχε την επιμέλεια και παρουσίαζε την εκπομπή ''Ρεπορτάζ χωρίς Σύνορα'' στη δημόσια τηλεόραση\n"
    "Input:  κατατάσσονται μετά απο διαγωνισμό Έλληνες πολίτες, πτυχιούχοι όλων των τμημάτων πανεπιστημιακών ή πολυτεχνικών σχολών ή κάτοχοι διπλώματος Ακαδημίας Εμπορικού Ναυτικού πλοιάρχων ή μηχανικών Correct grammatical or spelling errors in the given sentence.\nOutput:  κατατάσσονται μετά απο διαγωνισμό Έλληνες πολίτες, πτυχιούχοι συγκεκριμένων τμημάτων πανεπιστημιακών ή πολυτεχνικών σχολών ή κάτοχοι διπλώματος Ακαδημίας Εμπορικού Ναυτικού πλοιάρχων ή μηχανικών\n"
    "Input: Έτσι το σοβιετικό καθεστώς προσπαθεί απελπισμένα να θεωρείται ισότιμο με τις Ηνωμένες Πολιτείες Correct grammatical or spelling errors in the given sentence.\nOutput: Έτσι το σοβιετικό καθεστώς προσπαθεί απελπισμένα να θεωρείται ισότιμο από τις Ηνωμένες Πολιτείες\n"
    "Input:  Αθηνών-Πειραιώς μετείχε μεταξύ άλλων και η ομάδα Νεάπολη αλλά στον ημερήσιο τύπο της εποχής αναφέρεται ο σύλλογος Κεραυνός Πειραιώς Correct grammatical or spelling errors in the given sentence.\nOutput:  Αθηνών-Πειραιώς μετείχε μεταξύ άλλων και η ομάδα Νεάπολη αλλά στον ημερήσιπ τύπο της εποχής αναφέρεται ο σύλλογος Κεραυνός Πειραιώς\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_gr_30p = (
    "Input: Έκλειναν το head σε τενεκέ και χτυπούσαν με πλήκτρα να τρελάνουν τον ανακρινόμενο Correct grammatical or spelling errors in the given sentence.\nOutput: Μαλακιζονταν το head σε τενεκέ και χτυπούσαν με πλήκτρα να τρελάνουν τον ανακρινόμενο\n"
    "Input: Από το 1996 ο Κούλογλου επιμελούταν και παρουσίαζε την emission ''Ρεπορτάζ χωρίς Σύνορα'' στη public τηλεόραση Correct grammatical or spelling errors in the given sentence.\nOutput: Από το 1996 ο Κούλογλου είχε την επιμέλεια και παρουσίαζε την emission ''Ρεπορτάζ χωρίς Σύνορα'' στη public τηλεόραση\n"
    "Input:  κατατάσσονται after απο διαγωνισμό Έλληνες citizens, πτυχιούχοι όλων των τμημάτων πανεπιστημιακών ή πολυτεχνικών σχολών ή κάτοχοι διπλώματος Ακαδημίας Εμπορικού Ναυτικού πλοιάρχων ή μηχανικών Correct grammatical or spelling errors in the given sentence.\nOutput:  κατατάσσονται after απο διαγωνισμό Έλληνες citizens, πτυχιούχοι συγκεκριμένων τμημάτων πανεπιστημιακών ή πολυτεχνικών σχολών ή κάτοχοι διπλώματος Ακαδημίας Εμπορικού Ναυτικού πλοιάρχων ή μηχανικών\n"
    "Input: Έτσι το soviet καθεστώς προσπαθεί απελπισμένα να θεωρείται ισότιμο με τις Ηνωμένες states Correct grammatical or spelling errors in the given sentence.\nOutput: Έτσι το soviet καθεστώς προσπαθεί απελπισμένα να θεωρείται ισότιμο από τις Ηνωμένες states\n"
    "Input:  Αθηνών-Πειραιώς μετείχε μεταξύ άλλων και η ομάδα Νεάπολη αλλά στον ημερήσιο τύπο της εποχής αναφέρεται ο σύλλογος Κεραυνός Πειραιώς Correct grammatical or spelling errors in the given sentence.\nOutput:  Αθηνών-Πειραιώς μετείχε μεταξύ άλλων και η ομάδα Νεάπολη αλλά στον ημερήσιπ τύπο της εποχής αναφέρεται ο σύλλογος Κεραυνός Πειραιώς\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_gr_50p = (
    "Input: Έκλειναν το head σε τενεκέ και χτυπούσαν με πλήκτρα να τρελάνουν τον ανακρινόμενο Correct grammatical or spelling errors in the given sentence.\nOutput: Μαλακιζονταν το head σε τενεκέ και χτυπούσαν με πλήκτρα να τρελάνουν τον ανακρινόμενο\n"
    "Input: Από το 1996 ο Κούλογλου επιμελούταν και παρουσίαζε την emission ''Ρεπορτάζ χωρίς Σύνορα'' στη public television Correct grammatical or spelling errors in the given sentence.\nOutput: Από το 1996 ο Κούλογλου είχε την επιμέλεια και παρουσίαζε την emission ''Ρεπορτάζ χωρίς Σύνορα'' στη public television\n"
    "Input:  κατατάσσονται after απο contest Έλληνες citizens, πτυχιούχοι όλων των τμημάτων πανεπιστημιακών ή πολυτεχνικών σχολών ή κάτοχοι διπλώματος Ακαδημίας Εμπορικού Ναυτικού πλοιάρχων ή μηχανικών Correct grammatical or spelling errors in the given sentence.\nOutput:  κατατάσσονται after απο contest Έλληνες citizens, πτυχιούχοι συγκεκριμένων τμημάτων πανεπιστημιακών ή πολυτεχνικών σχολών ή κάτοχοι διπλώματος Ακαδημίας Εμπορικού Ναυτικού πλοιάρχων ή μηχανικών\n"
    "Input: Έτσι το soviet regime προσπαθεί απελπισμένα να θεωρείται ισότιμο με τις Ηνωμένες states Correct grammatical or spelling errors in the given sentence.\nOutput: Έτσι το soviet regime προσπαθεί απελπισμένα να θεωρείται ισότιμο από τις Ηνωμένες states\n"
    "Input:  Αθηνών-Πειραιώς μετείχε μεταξύ άλλων και η ομάδα Νεάπολη αλλά στον ημερήσιο τύπο της εποχής αναφέρεται ο σύλλογος lightning piraeus Correct grammatical or spelling errors in the given sentence.\nOutput:  Αθηνών-Πειραιώς μετείχε μεταξύ άλλων και η ομάδα Νεάπολη αλλά στον ημερήσιπ τύπο της εποχής αναφέρεται ο σύλλογος lightning piraeus\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_gr_80p = (
    "Input: Έκλειναν το head σε τενεκέ και χτυπούσαν με keyboard να τρελάνουν τον ανακρινόμενο Correct grammatical or spelling errors in the given sentence.\nOutput: Μαλακιζονταν το head σε τενεκέ και χτυπούσαν με keyboard να τρελάνουν τον ανακρινόμενο\n"
    "Input: Από το 1996 ο Κούλογλου επιμελούταν και παρουσίαζε την emission ''Ρεπορτάζ χωρίς frontier'' στη public television Correct grammatical or spelling errors in the given sentence.\nOutput: Από το 1996 ο Κούλογλου είχε την επιμέλεια και παρουσίαζε την emission ''Ρεπορτάζ χωρίς frontier'' στη public television\n"
    "Input:  κατατάσσονται after απο contest Έλληνες citizens, πτυχιούχοι όλων των τμημάτων πανεπιστημιακών ή πολυτεχνικών σχολών ή κάτοχοι διπλώματος academy Εμπορικού Ναυτικού πλοιάρχων ή μηχανικών Correct grammatical or spelling errors in the given sentence.\nOutput:  κατατάσσονται after απο contest Έλληνες citizens, πτυχιούχοι συγκεκριμένων τμημάτων πανεπιστημιακών ή πολυτεχνικών σχολών ή κάτοχοι διπλώματος academy Εμπορικού Ναυτικού πλοιάρχων ή μηχανικών\n"
    "Input: Έτσι το soviet regime προσπαθεί desperately να θεωρείται ισότιμο με τις Ηνωμένες states Correct grammatical or spelling errors in the given sentence.\nOutput: Έτσι το soviet regime προσπαθεί desperately να θεωρείται ισότιμο από τις Ηνωμένες states\n"
    "Input:  Αθηνών-Πειραιώς μετείχε μεταξύ άλλων και η team Νεάπολη αλλά στον ημερήσιο τύπο της εποχής αναφέρεται ο σύλλογος lightning piraeus Correct grammatical or spelling errors in the given sentence.\nOutput:  Αθηνών-Πειραιώς μετείχε μεταξύ άλλων και η team Νεάπολη αλλά στον ημερήσιπ τύπο της εποχής αναφέρεται ο σύλλογος lightning piraeus\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_hi_gec = (
    "Input: अर्थात , हिमालय से प्रारम्भ होकर इन्दु सरोवर ( हिन्द महासागर ) तक यह देव निर्मित देश हिन्दुस्थान कहलाता था . Correct grammatical or spelling errors in the given sentence.\nOutput: अर्थात , हिमालय से प्रारम्भ होकर इन्दु सरोवर ( हिन्द महासागर ) तक यह देव निर्मित देश हिन्दुस्थान कहलाता है .\n"
    "Input: जन्म 19 मई 1910 मृत्यु 15 नवम्बर 1949 ) एक पत्रकार , हिन्दू राष्ट्रवादी थे . Correct grammatical or spelling errors in the given sentence.\nOutput: जन्म 19 मई 1910 मृत्यु 15 नवम्बर 1949 ) एक पत्रकार , हिन्दू राष्ट्रवादी था .\n"
    "Input: सन् १९६९ में गढ़वाल मण्डल की स्थापना की गई जिसका मुख्यालय पौड़ी बनाया गया . Correct grammatical or spelling errors in the given sentence.\nOutput: सन् १९६९ में गढ़वाल मण्डल की स्थापना की गयी जिसका मुख्यालय पौड़ी बनाया गया .\n"
    "Input: सैय्यद कासिम हसन , भारत के उत्तर प्रदेश की सोलहवीं विधानसभा सभा में विधायक हैं . Correct grammatical or spelling errors in the given sentence.\nOutput: सैय्यद कासिम हसन , भारत के उत्तर प्रदेश की सोलहवीं विधानसभा सभा में विधायक रहे .\n"
    "Input: यही कारण है कि क्रिश्चियन संस्थाओं में अध्ययन करते हुए राधाकृष्णन के जीवन में उच्च गुण समाहित हो गए . Correct grammatical or spelling errors in the given sentence.\nOutput: यही कारण है कि क्रिश्चियन संस्थाओं में अध्ययन करते हुए राधाकृष्णन के जीवन में उच्च गुण समाहित हो गये .\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_hi_gec_30p = (
    "Input: अर्थात , हिमालय से start होकर इन्दु सरोवर ( हिन्द महासागर ) तक यह देव built देश हिन्दुस्थान कहलाता था . Correct grammatical or spelling errors in the given sentence.\nOutput: अर्थात , हिमालय से start होकर इन्दु सरोवर ( हिन्द महासागर ) तक यह देव built देश हिन्दुस्थान कहलाता है .\n"
    "Input: Birth 19 मई 1910 मृत्यु 15 नवम्बर 1949 ) एक पत्रकार , हिन्दू nationalist थे . Correct grammatical or spelling errors in the given sentence.\nOutput: Birth 19 मई 1910 मृत्यु 15 नवम्बर 1949 ) एक पत्रकार , हिन्दू nationalist था .\n"
    "Input: सन् १९६९ में गढ़वाल मण्डल की establishment की गई जिसका मुख्यालय पौड़ी बनाया गया . Correct grammatical or spelling errors in the given sentence.\nOutput: सन् १९६९ में गढ़वाल मण्डल की establishment की गयी जिसका मुख्यालय पौड़ी बनाया गया .\n"
    "Input: सैय्यद कासिम हसन , भारत के Uttar Pradesh की सोलहवीं विधानसभा सभा में विधायक हैं . Correct grammatical or spelling errors in the given sentence.\nOutput: सैय्यद कासिम हसन , भारत के Uttar Pradesh की सोलहवीं विधानसभा सभा में विधायक रहे .\n"
    "Input: यही कारण है कि christian संस्थाओं में study करते हुए राधाकृष्णन के जीवन में उच्च गुण समाहित हो गए . Correct grammatical or spelling errors in the given sentence.\nOutput: यही कारण है कि christian संस्थाओं में study करते हुए राधाकृष्णन के जीवन में उच्च गुण समाहित हो गये .\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_hi_gec_50p = (
    "Input: अर्थात , हिमालय से start होकर इन्दु सरोवर ( हिन्द महासागर ) तक यह देव built country हिन्दुस्थान कहलाता था . Correct grammatical or spelling errors in the given sentence.\nOutput: अर्थात , हिमालय से start होकर इन्दु सरोवर ( हिन्द महासागर ) तक यह देव built country हिन्दुस्थान कहलाता है .\n"
    "Input: Birth 19 मई 1910 death 15 नवम्बर 1949 ) एक पत्रकार , हिन्दू nationalist थे . Correct grammatical or spelling errors in the given sentence.\nOutput: Birth 19 मई 1910 death 15 नवम्बर 1949 ) एक पत्रकार , हिन्दू nationalist था .\n"
    "Input: सन् १९६९ में गढ़वाल मण्डल की establishment की गई जिसका headquarters पौड़ी बनाया गया . Correct grammatical or spelling errors in the given sentence.\nOutput: सन् १९६९ में गढ़वाल मण्डल की establishment की गयी जिसका headquarters पौड़ी बनाया गया .\n"
    "Input: सैय्यद कासिम हसन , भारत के Uttar Pradesh की सोलहवीं विधानसभा सभा में legislator हैं . Correct grammatical or spelling errors in the given sentence.\nOutput: सैय्यद कासिम हसन , भारत के Uttar Pradesh की सोलहवीं विधानसभा सभा में legislator रहे .\n"
    "Input: यही reason है कि christian संस्थाओं में study करते हुए राधाकृष्णन के जीवन में उच्च गुण समाहित हो गए . Correct grammatical or spelling errors in the given sentence.\nOutput: यही reason है कि christian संस्थाओं में study करते हुए राधाकृष्णन के जीवन में उच्च गुण समाहित हो गये .\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )

    prompt_hi_gec_80p = (
    "Input: अर्थात , हिमालय से start होकर इन्दु सरोवर ( हिन्द महासागर ) तक यह देव built country हिन्दुस्थान called था . Correct grammatical or spelling errors in the given sentence.\nOutput: अर्थात , हिमालय से start होकर इन्दु सरोवर ( हिन्द महासागर ) तक यह देव built country हिन्दुस्थान called है .\n"
    "Input: Birth 19 मई 1910 death 15 नवम्बर 1949 ) एक journalist , हिन्दू nationalist थे . Correct grammatical or spelling errors in the given sentence.\nOutput: Birth 19 मई 1910 death 15 नवम्बर 1949 ) एक journalist , हिन्दू nationalist था .\n"
    "Input: year १९६९ में गढ़वाल मण्डल की establishment की गई जिसका headquarters पौड़ी बनाया गया . Correct grammatical or spelling errors in the given sentence.\nOutput: year १९६९ में गढ़वाल मण्डल की establishment की गयी जिसका headquarters पौड़ी बनाया गया .\n"
    "Input: सैय्यद कासिम हसन , भारत के Uttar Pradesh की सोलहवीं assembly सभा में legislator हैं . Correct grammatical or spelling errors in the given sentence.\nOutput: सैय्यद कासिम हसन , भारत के Uttar Pradesh की सोलहवीं assembly सभा में legislator रहे .\n"
    "Input: यही reason है कि christian संस्थाओं में study करते हुए राधाकृष्णन के life में उच्च गुण contained हो गए . Correct grammatical or spelling errors in the given sentence.\nOutput: यही reason है कि christian संस्थाओं में study करते हुए राधाकृष्णन के life में उच्च गुण contained हो गये .\n"
    f"Input: {text_input} Correct grammatical or spelling errors in the given sentence.\nOutput: "
    )


    prompt_eng_3 = f"Input: I umm.. don't want to go uhh.. to the school. Remove disfluencies in the given sentence.\nOutput: I don't want to go to the school.\nInput: The sun sun rises in the east. Remove disfluencies in the given sentence.\nOutput: The sun rises in the east.\nInput: There are nin eight planets in the solar system. Remove disfluencies in the given sentence.\nOutput: There are eight planets in the solar system.\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_prompt_3 = f"Input: अ मुझे बताइए ये सेवा नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे बताइए ये सेवा नहीं है क्या?\nInput: लेकिन यहां पर पर अटकने से काम नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन यहां पर अटकने से काम नहीं होगा।\nInput: बाजपेयी भी इन अर अर्थशास्त्रियों में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी भी इन अर्थशास्त्रियों में शामिल थे।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_30p_3 = f"Input: अ मुझे बताइए ये service नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे बताइए ये service नहीं है क्या?\nInput: लेकिन यहां पर पर अटकने से work नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन यहां पर अटकने से work नहीं होगा।\nInput: बाजपेयी also इन अर अर्थशास्त्रियों में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी also इन अर्थशास्त्रियों में शामिल थे।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_50p_3 = f"Input: अ मुझे बताइए ये service नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे बताइए ये service नहीं है क्या?\nInput: लेकिन here पर पर अटकने से work नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन here पर अटकने से work नहीं होगा।\nInput: बाजपेयी भी इन अर economists में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी भी इन economists में शामिल थे।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "
    hindi_80p_3 = f"Input: अ मुझे tell ये service नहीं है क्या? Remove disfluencies in the given sentence.\nOutput: मुझे tell ये service नहीं है क्या?\nInput: लेकिन here पर पर cling से work नहीं होगा। Remove disfluencies in the given sentence.\nOutput: लेकिन here पर cling से work नहीं होगा।\nInput: बाजपेयी also इन अर economists में शामिल थे। Remove disfluencies in the given sentence.\nOutput: बाजपेयी also इन economists में शामिल थे।\nInput: {text_input} Remove disfluencies in the given sentence.\nOutput: "


    
    prompt = prompt_eng
    # temp = sys.argv[1]
    # prompt = locals().get(temp, "Variable not found!")


    input_ids = tokenizer(prompt, return_tensors="pt").to(0)
    sample = model.generate(**input_ids, max_length=1500,
                            top_k=0, temperature=0.0)

    # output = tokenizer.decode(sample[0])
    # print(output)

    output = tokenizer.decode(sample[0]).split("<pad>")[1].split("</s>")[0]

    # if(tokenizer.decode(sample[0])[0] != '<pad>'):
    #     output = tokenizer.decode(sample[0]).split("</s>")[0] 
    # else:
    #     output = tokenizer.decode(sample[0]).split("<pad>")[1].split("</s>")[0] #for mT0
    # output = tokenizer.decode(sample[0]).split("</s>")[0] #for bloomz
    # output = tokenizer.decode(sample[0])
    output = output.split("Output: ")[-1]
    print(output)
    
    instance.append(output)
    # instance.append(i[2])
    resultList.append(instance)

    if count % 10 == 0:
        print("{} instances done".format(count))

# filename = 'MT0_xxl_results/result_g_80p.csv'
filename = f'MT0_xxl_results/{sys.argv[2]}'
fields = ['disfluent', 'label', 'pred_label']

with open(filename, 'w',encoding="utf-8") as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(resultList)