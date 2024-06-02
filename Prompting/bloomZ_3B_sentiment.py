import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # SET the GPUs you want to use
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd


# Following line makes the model load in memory
torch.set_default_tensor_type(torch.cuda.FloatTensor)

model = AutoModelForCausalLM.from_pretrained(
    "models/bloomz-3b", use_cache=True)
tokenizer = AutoTokenizer.from_pretrained("models/bloomz-3b")

# model = AutoModelForCausalLM.from_pretrained(
#     "bigscience/bloomz-3b", use_cache=True)
# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-3b")

# model = AutoModelForCausalLM.from_pretrained(
#     "facebook/xglm-2.9B", use_cache=True
# tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-2.9B")

# Following line for reproducibility
set_seed(2023)
# set_seed(42)

testFile = open('hi_sentiment_test.csv', 'r')
reader = csv.reader(testFile)
testList = []
for instance in reader:
    testList.append(instance)

resultList = []
predictions = []
ground_truth = []
count = 0

label_map = {'0': 'neutral', '1': 'positive', '-1': 'negative', 'neutral':'neutral', 'positive':'positive', 'negative':'negative'}

for i in testList[1:]:

    count += 1
    instance = []
    
    
    text_input = i[1]
    correct_label = label_map[i[0].strip()]
    instance.append(correct_label)
    instance.append(i[1])

    
    prompt_eng = (
        "Input: India's argument is that there is no need to sign the treaty as it separates the nuclear program into civilian and military purposes. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: It is advised to provide all possible support to the people. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: It is not possible to change many laws according to the needs. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        "Input: After that, they discussed the programs to be undertaken by regions. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: Enrollment of TRS membership is in full swing. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: Whatever the reason, none of the Telugu states got a ministerial position at the Centre. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        f"Input: {text_input}, What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: "
    )

    prompt_tel = (
        "Input: అణు కార్యక్రమాన్ని పౌర అవసరాలు, సైనిక అవసరాలుగా విడదీసినందున ఆ ఒప్పందంపై సంతకం పెట్టాల్సిన అవసరం లేదని భారత్ వాదన. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: ప్రజలకు అన్నివిధాలా తోడ్పాటును అందించాలని సూచించారు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: అవసరాలకు అనుగుణంగా అనేక చట్టాలను మార్చుకోవడం సాధ్యం కావడం లేదు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        "Input: అనంతరం ప్రాంతాల వారీగా చేపట్టే కార్యక్రమాలపై చర్చించారు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: జోరుగా టీఆరెస్ సభ్యత్వ నమోదు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: కారణం ఏమిటో గానీ తెలుగు రాష్ట్రాల్లో ఏ ఒక్కరికీ కేంద్రంలో మంత్రిపదవి దక్కలేదు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        f"Input: {text_input}, What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: "
    )

    prompt_t_mixed30 = (
        "Input: అణు program పౌర అవసరాలు, సైనిక needs విడదీసినందున ఆ contractపై సంతకం పెట్టాల్సిన అవసరం లేదని భారత్ వాదన. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: people అన్నివిధాలా తోడ్పాటును అందించాలని సూచించారు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: అవసరాలకు accordingగా అనేక చట్టాలను మార్చుకోవడం సాధ్యం కావడం లేదు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        "Input: అనంతరం regions వారీగా చేపట్టే కార్యక్రమాలపై చర్చించారు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: జోరుగా టీఆరెస్ సభ్యత్వ registration. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: reason ఏమిటో గానీ తెలుగు రాష్ట్రాల్లో ఏ ఒక్కరికీ కేంద్రంలో మంత్రిపదవి దక్కలేదు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        f"Input: {text_input}, What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: "
    )

    prompt_t_mixed50 = (
        "Input: అణు program పౌర అవసరాలు, సైనిక needsగా విడదీసినందున ఆ contractపై signature put అవసరం లేదని భారత్ వాదన. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: people అన్నివిధాలా తోడ్పాటును అందించాలని suggested. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: అవసరాలకు accordingగా many lawsను మార్చుకోవడం సాధ్యం కావడం లేదు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        "Input: Later regions వారీగా చేపట్టే కార్యక్రమాలపై చర్చించారు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: జోరుగా టీఆరెస్ membership registration. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: reason ఏమిటో గానీ తెలుగు statesల్లో ఏ ఒక్కరికీ centre ministry దక్కలేదు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        f"Input: {text_input}, What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: "
    )

    prompt_t_mixed80 = (
        "Input: అణు program పౌర అవసరాలు, military needsగా విడదీసినందున ఆ contractపై signature put need లేదని భారత్ argument. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: people అన్నివిధాలా supportను అందించాలని suggested. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: అవసరాలకు accordingగా many lawsను change possible కావడం లేదు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        "Input: Later regions వారీగా చేపట్టే programsపై discussed. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: Hurryగా టీఆరెస్ membership registration. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: reason ఏమిటో గానీ తెలుగు statesల్లో ఏ ఒక్కరికీ centre ministry దక్కలేదు. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        f"Input: {text_input}, What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: "
    )

    prompt_h_eng = (
        "Input: Rhinoceros as well as elephants, cheetahs, tigers, deer, dolphins, sambar etc. can be seen in this park located on the banks of the Brahmaputra river in Assam. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: Its 13 mega pixel camera which is the hero of this device. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: Overall, the performance of the G3 Stylus cannot be called good. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        "Input: The 3120 mAh battery inside it lasts for a full day and a half. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: Its benchmark scores were very promising as it blows away the smartphone competition. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: The disadvantage of which is that if you are watching a movie or playing a game, then the speakers get covered by your hands. What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        f"Input: {text_input}, What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: "
    )

    prompt_hin = (
        "Input: असम में ब्रह्मपुत्र नदी के किनारे स्थित इस पार्क में गैंडे के साथ - साथ हाथी , चीता , बाघ , हिरण , डॉल्फिन , सांभर आदि देखे जा सकते हैं । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: इसका 13 मेगा पिक्सल कैमरा जो इस डिवाईस का हिरो है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: कुल मिलाकर जी3 स्टाइलस का परफॉर्मेंस अच्छा नहीं कहा जा सकता । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        "Input: इसके अन्दर लगी 3120 एमएएच की बैटरी , पूरे डेढ़ दिन तक चलती है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: इसके बेंचमार्क स्कोर्स बहुत ही आशाजनक थे क्योंकि यह स्मार्टफोन प्रतियोगिता को दूर कर देता है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: जिसका नुकसान ये होता है कि अगर आप फिल्म देख रहे है या गेम खेल रहे है तो स्पीकर्स आपके हाथों से ढक जाते हैं । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        f"Input: {text_input}, What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: "
    )

    prompt_h_mixed30 = (
        "Input: असम में ब्रह्मपुत्र river के किनारे स्थित इस park में गैंडे के साथ - साथ हाथी , चीता , बाघ , हिरण , डॉल्फिन , सांभर आदि see जा सकते हैं । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: इसका 13 mega पिक्सल camera जो इस डिवाईस का हिरो है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: total मिलाकर जी3 स्टाइलस का परफॉर्मेंस good नहीं कहा जा सकता । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        "Input: इसके inside लगी 3120 एमएएच की बैटरी , पूरे डेढ़ days तक चलती है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: इसके बेंचमार्क scores बहुत ही आशाजनक थे क्योंकि यह स्मार्टफोन प्रतियोगिता को दूर कर देता है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: जिसका damage ये होता है कि अगर आप फिल्म see रहे है या गेम खेल रहे है तो स्पीकर्स आपके हाथों से ढक जाते हैं । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        f"Input: {text_input}, What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: "
    )

    prompt_h_mixed50 = (
        "Input: असम में ब्रह्मपुत्र river के bank स्थित इस park में गैंडे के साथ - साथ elephant , cheetah , बाघ , हिरण , डॉल्फिन , सांभर आदि see जा सकते हैं । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: इसका 13 mega pixel camera जो इस डिवाईस का हिरो है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: total मिलाकर जी3 stylus का परफॉर्मेंस good नहीं कहा जा सकता । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        "Input: इसके inside लगी 3120 एमएएच की battery , पूरे डेढ़ days तक चलती है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: इसके benchmark scores बहुत ही hopeful थे क्योंकि यह स्मार्टफोन प्रतियोगिता को दूर कर देता है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: जिसका damage ये होता है कि अगर आप film देख रहे है या game खेल रहे है तो स्पीकर्स आपके हाथों से ढक जाते हैं । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        f"Input: {text_input}, What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: "
    )

    prompt_h_mixed80 = (
        "Input: असम में ब्रह्मपुत्र river के bank situated इस park में गैंडे के साथ - साथ elephant , cheetah , बाघ , deer , dolphin , सांभर आदि see जा सकते हैं । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: इसका 13 mega pixel camera जो इस device का हिरो है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: total मिलाकर जी3 stylus का performance good नहीं कहा जा सकता । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        "Input: इसके inside लगी 3120 एमएएच की battery , पूरे डेढ़ days तक go है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: neutral\n"
        "Input: इसके benchmark scores बहुत ही hopeful थे क्योंकि यह smartphone competition को दूर कर देता है । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: positive\n"
        "Input: जिसका damage ये होता है कि अगर आप film देख रहे है या game play रहे है तो speakers आपके हाथों से ढक जाते हैं । What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: negative\n"
        f"Input: {text_input}, What is the sentiment of the given sentence? postive, neutral, or negative?\nOutput: "
    )
    
    prompt = prompt_h_mixed80





    input_ids = tokenizer(prompt, return_tensors="pt").to(0)
    sample = model.generate(**input_ids, max_length=1600,
                            top_k=0, temperature=0.0)


    if(tokenizer.decode(sample[0])[0] != '<pad>'):
        output = tokenizer.decode(sample[0]).split("</s>")[0] 
    else:
        output = tokenizer.decode(sample[0]).split("<pad>")[1].split("</s>")[0] #for mT0
   
    output = output.split("Output: ")[-1]
    prediction = output.strip().lower()
    
    predictions.append(prediction)
    ground_truth.append(correct_label)
    print(prediction)
    
    instance.append(output)
    # instance.append(i[2])
    resultList.append(instance)

    if count % 10 == 0:
        print("{} instances done".format(count))

macro_precision, macro_recall, macro_fscore, _ = precision_recall_fscore_support(
    ground_truth, predictions, labels=['neutral', 'positive', 'negative'], 
    average='macro', zero_division=0
)

# Compute accuracy
accuracy = sum([ground_truth[i] == predictions[i] for i in range(len(predictions))]) / len(predictions)

# Compute per-class metrics
precision, recall, fscore, support = precision_recall_fscore_support(
    ground_truth, predictions, labels=['neutral', 'positive', 'negative'], 
    zero_division=0
)

# Create a pandas DataFrame for representation
df = pd.DataFrame({
    'Label': ['neutral', 'positive', 'negative', 'macro-average'],
    'Precision': list(precision) + [macro_precision],
    'Recall': list(recall) + [macro_recall],
    'F-Score': list(fscore) + [macro_fscore],
    'Support': list(support) + ['N/A'],
    'Accuracy': list(['N/A'] * 3) + [accuracy]
})

print(df)


filename = 'BloomZ_3B_sentiment/result_h_mixed80.csv'
fields = ['label', 'sentence', 'pred_label']

with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(resultList)
