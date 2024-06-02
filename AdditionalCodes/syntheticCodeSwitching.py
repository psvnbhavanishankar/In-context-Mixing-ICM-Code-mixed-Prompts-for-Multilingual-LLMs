import random
import spacy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # SET the GPUs you want to use

# Load the NLLB model for translation
#tel_Telu
#hin_Deva
#mar_Deva
#ben_Beng
#vie_Latn
#ces_Latn
#por_Latn
tokenizer = AutoTokenizer.from_pretrained("models/nllb-200-3.3B", src_lang="por_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("models/nllb-200-3.3B")

def get_translation(word):
    """Fetch the English translation for a given Telugu word using the NLLB model."""
    inputs = tokenizer(word, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=100
    )
    english_phrase = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    return english_phrase

def fetch_translation(word, content=1):
    """Fetch the English translation for a given Telugu word using the NLLB model."""
    inputs = tokenizer(word, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=100
    )
    english_phrase = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    if content:
        # Extract content words from the translated phrase
        content_words = [token.text for token in nlp_en(english_phrase) if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]
        
        # Return the first content word, or the whole phrase if no content words are found
        return content_words[0] if content_words else english_phrase

    return english_phrase

def get_pos_tag_english(word):
    """Get the POS tag of the translated English word using spaCy."""
    doc = nlp_en(word)
    return doc[0].pos_

def code_switch(sentence, ratio=0.5, content=1):
    """Perform code switching based on the given ratio."""
    words = sentence.split()
    new_sentence = []

    for word in words:
        english_word = fetch_translation(word)
        if content:
            pos_tag = get_pos_tag_english(english_word)
            if pos_tag in ["NOUN", "VERB", "ADJ", "ADV"] and random.random() < ratio:
                new_sentence.append(english_word)
            else:
                new_sentence.append(word)

        else:
            if random.random() < ratio:
                new_sentence.append(english_word)
            else:
                new_sentence.append(word)

    return ' '.join(new_sentence)

# Load spaCy's English model for POS tagging
nlp_en = spacy.load("en_core_web_sm")

# Test
telugu_sentences = ['నేను ఉహుహు... ప్రపంచ కప్ మ్యాచ్ చూడాలనుకుంటున్నాను','అరేరే నేను నా మొబైల్ ఎక్కడ పెట్టానో మరచిపోయాను!', 'ఆయన బాగా బాగా ఆడుతున్నాడు క్రికెట్.', 'మన ప్రయాణం శుక్రవా శనివారం నాడు కాదండి?', '3:30 కాదు కాదు, 4:30 కి మన ప్రయాణం.', 'నా పుస్తకం నాకు పేపర్ కావాలి.']
hindi_sentences = ['अ मुझे बताइए ये सेवा नहीं है क्या?', 'लेकिन यहां पर पर अटकने से काम नहीं होगा।', 'बाजपेयी भी इन अर अर्थशास्त्रियों में शामिल थे।', 'मैंने सोचा कि, उम्म...क्या मैं आज शाम को फिल्म देखने जाऊँ?', 'अरे, यह कुत्ता हमारे पास क्यों आ रहा है?', 'क्या हमें कल.. हमें कल चलना चाहिए।', 'उसके...मतलब, उसने अपनी नई कार खरीदी है।']
marathi_sentences = ['अ मला सांगा, ही सेवा नाही का?', 'माझ्या माझ्या कामाची चर्चा आहे उद्या.', 'या अर अर्थतज्ज्ञांमध्ये वाजपेयींचाही समावेश होता.', 'मी विचार केला कि, अं...मी आज सायंकाळी मी चित्रपट पहायला जाऊ का?', 'अरे हा कुत्रा आमच्याजवळ का येतोय?', 'का आपण उद्या.. आपण उद्या जायला हवं.', 'त्याची...म्हणजे, त्याने त्याची नवीन खरेदी केली आहे.']
bengali_sentences = ['আহ আমাকে বলুন, এটা কি পরিষেবা নয়?', 'কিন্তু কিন্তু এখানে আটকালে তো কাজ হবেনা।', 'বাজপেয়ী জিও এইসব অর্থ অর্থনীতি তে অন্তর্ভুক্ত ছিলেন।', 'আমি ভাবলাম কি যে, আঃ আজকে সন্ধায় কি আমি সিনেমা দেখতে যাব?', 'আরেহ এই কুকুর টা আমাদের দিকে কেন আসছে?', 'আমিাদের কি কাল, আমাদের কাল যাওয়া উচিত।', 'ওনার, মানে উনি নিজের নতুন গাড়ি কিনেছেন।']
viet_sentences = ['tôi cần thuê à tôi muốn bay một chuyến khứ hồi từ đà nẵng đến đà lạt.', 'sân bay ừm không hãng hàng không nào có đường bay từ bắc kinh ờ ý tôi là thượng thượng hải đến washington dc mà cần nối chuyến qua các thành phố khác.', 'cho tôi biết tất cả các máy bay à chuyến bay từ huế đến quy nhơn.', 'đà nẵng đến ờ hồ chí minh í lộn đến cà mau.', 'có bao nhiêu ghế à ý tôi là hạng ghế.', 'chuyến bay nào rời buôn ma thuột vào ngày mùng 4 ờ không ngày 5 tháng 7 sau 7 giờ tối và đến cần thơ.']
czech_sentences = ['Strávily jsme měsíc v hlavním městě Jemenu Sané , kde jsme se zúčastnily kurzu arabštiny.', 'Musíme být úspěšní poprvé sámi.']


#XNLI
telugu_xnli = ['మా నంబర్‌లో ఒకరు మీ సూచనలను సూక్ష్మంగా అమలు చేస్తారు.', 'నా బృందంలోని సభ్యుడు మీ ఆర్డర్‌లను చాలా ఖచ్చితత్వంతో అమలు చేస్తారు.', 'స్వలింగ సంపర్కులు మరియు లెస్బియన్లు.', 'భిన్న లింగ సంపర్కులు.', 'వేద వైపు తిరిగి నవ్వాడు.', 'తల్లితో కలిసి తన వెనకే మెల్లగా నడుస్తున్న వేదను చూసి నవ్వాడు.', 'నీకు ఎలా తెలుసు ? ఇదంతా మళ్లీ వారి సమాచారం.', 'ఈ సమాచారం వారికే చెందుతుంది.', 'జాతీయ ఉద్యానవనాలు మరియు నిర్జన ప్రాంతాలలో సహజ పరిస్థితులకు తిరిగి రావాలనే కాంగ్రెస్ నిర్దేశించిన లక్ష్యం వైపు రాష్ట్రాలు తమ రాష్ట్ర అమలు ప్రణాళికలలో సహేతుకమైన పురోగతిని చూపాలి.', 'ఏదైనా మెరుగుదల ఉండాల్సిన అవసరం లేదు.', 'ఆమె తిరిగి నవ్వింది.', 'ఆమె నవ్వు ఆపుకోలేక చాలా సంతోషించింది.']
hindi_xnli = ['हमारा एक नंबर आपके निर्देशों का सूक्ष्मता से पालन करेगा।', 'मेरी टीम का एक सदस्य आपके आदेशों को अत्यंत सटीकता के साथ निष्पादित करेगा।', 'समलैंगिक और लेस्बियन।', 'विषमलैंगिक।', 'वह मुड़ा और वेदा की ओर देखकर मुस्कुराया।', 'वह वेदा को देखकर मुस्कुराया जो अपनी माँ के साथ उसके पीछे धीरे-धीरे चल रही थी।', 'आपको कैसे मालूम ? ये सब उनकी जानकारी है।', 'ये जानकारी उनकी है।', 'राज्यों को राष्ट्रीय उद्यानों और जंगल क्षेत्रों में प्राकृतिक परिस्थितियों में लौटने के कांग्रेस द्वारा निर्धारित लक्ष्य की दिशा में अपनी राज्य कार्यान्वयन योजनाओं में उचित प्रगति दिखानी चाहिए।', 'इसमें कोई सुधार होना जरूरी नहीं है।', 'वह वापस मुस्कुराई।', 'वह इतनी खुश थी कि वह मुस्कुराना बंद नहीं कर पा रही थी।']
marathi_xnli = ['आमचा एक नंबर तुमच्या सूचनांची काटेकोरपणे अंमलबजावणी करेल.', 'माझ्या टीमचा एक सदस्य तुमच्या ऑर्डर्स अत्यंत अचूकतेने अंमलात आणेल.', 'समलिंगी आणि समलैंगिक.', 'भिन्नलिंगी.', 'तो वळून वेदाकडे हसला.', 'तो वेदाकडे बघून हसला जो आईसोबत त्याच्या मागे हळू चालत होता.', 'तुला कसे माहीत ? ही सर्व त्यांची माहिती आहे.', 'ही माहिती त्यांच्या मालकीची आहे.', 'राष्ट्रीय उद्याने आणि वाळवंट भागात नैसर्गिक परिस्थितीत परत येण्याच्या कॉंग्रेसने अनिवार्य केलेल्या उद्दिष्टाच्या दिशेने राज्यांनी त्यांच्या राज्य अंमलबजावणी योजनांमध्ये वाजवी प्रगती दर्शविली पाहिजे.', 'त्यात काही सुधारणा होणे आवश्यक नाही.', 'ती परत हसली.', 'तिला इतका आनंद झाला होता की तिला हसू आवरता आले नाही.']
bengali_xnli = ['আমাদের নম্বরগুলির মধ্যে একটি আপনার নির্দেশাবলী মিনিটে কার্যকর করবে।', 'আমার দলের একজন সদস্য আপনার আদেশগুলি অত্যন্ত নির্ভুলতার সাথে কার্যকর করবে।', 'সমকামী এবং সমকামীরা।', 'বিষমকামী।', 'সে ঘুরে বেদের দিকে তাকিয়ে হাসল।', 'সে বেদাকে দেখে হাসল যে তার মায়ের সাথে তার পিছনে ধীরে ধীরে হাঁটছিল।', 'তুমি কিভাবে জান ? এসবই তাদের তথ্য।', 'এই তথ্য তাদের অন্তর্গত।', 'জাতীয় উদ্যান এবং মরুভূমি অঞ্চলে প্রাকৃতিক পরিস্থিতিতে ফিরে আসার কংগ্রেসের নির্দেশিত লক্ষ্যের দিকে রাজ্যগুলিকে অবশ্যই তাদের রাষ্ট্রীয় বাস্তবায়ন পরিকল্পনায় যুক্তিসঙ্গত অগ্রগতি দেখাতে হবে।', 'এর জন্য কোনো উন্নতির প্রয়োজন নেই।', 'সে ফিরে হাসল।', 'সে এত খুশি ছিল যে সে হাসি থামাতে পারেনি।']

#Sentiment
telugu_sentiment = ['అణు కార్యక్రమాన్ని పౌర అవసరాలు, సైనిక అవసరాలుగా విడదీసినందున ఆ ఒప్పందంపై సంతకం పెట్టాల్సిన అవసరం లేదని భారత్ వాదన.', 'ప్రజలకు అన్నివిధాలా తోడ్పాటును అందించాలని సూచించారు.', 'అవసరాలకు అనుగుణంగా అనేక చట్టాలను మార్చుకోవడం సాధ్యం కావడం లేదు.', 'అనంతరం ప్రాంతాల వారీగా చేపట్టే కార్యక్రమాలపై చర్చించారు.', 'జోరుగా టీఆరెస్ సభ్యత్వ నమోదు.', 'కారణం ఏమిటో గానీ తెలుగు రాష్ట్రాల్లో ఏ ఒక్కరికీ కేంద్రంలో మంత్రిపదవి దక్కలేదు.']
hindi_sentiment = ['असम में ब्रह्मपुत्र नदी के किनारे स्थित इस पार्क में गैंडे के साथ - साथ हाथी , चीता , बाघ , हिरण , डॉल्फिन , सांभर आदि देखे जा सकते हैं ।', 'इसका 13 मेगा पिक्सल कैमरा जो इस डिवाईस का हिरो है ।', 'कुल मिलाकर जी3 स्टाइलस का परफॉर्मेंस अच्छा नहीं कहा जा सकता ।', 'इसके अन्दर लगी 3120 एमएएच की बैटरी , पूरे डेढ़ दिन तक चलती है ।', 'इसके बेंचमार्क स्कोर्स बहुत ही आशाजनक थे क्योंकि यह स्मार्टफोन प्रतियोगिता को दूर कर देता है ।', 'जिसका नुकसान ये होता है कि अगर आप फिल्म देख रहे है या गेम खेल रहे है तो स्पीकर्स आपके हाथों से ढक जाते हैं ।']
# marathi_sentiment = 
# bengali_sentiment = 

hindi_qa = ['पैंथर्स डिफ़ेंस ने कितने अंक दिए?', 'डिवीजनल राउंड में ब्रोंकोस से कौन हारा?', 'वर्तमान में ब्रॉनकोस फ्रैंचाइज़ी में जॉन एलवे की क्या भूमिका है?', 'लेडी गागा ने कितने ग्रैमी जीते हैं?']
marathi_ape = ['हळूहळू खायला आणि प्यायला मदत होते आणि लहान, वारंवार जेवण होते.', 'कधी कधी खांद्यावरून बाहेर पडणाऱ्या आगीचे चित्रण केले जात नाही.', 'पिंपळाच्या आकाराचे मातीचे शरीर, संपूर्ण शरीरावर लाल कापड चिकटवले जाते.', 'या कालखंडात आणखी एक महत्त्वाची गोष्ट म्हणजे तांत्रिकवादाची वाढ.', 'या कामांच्या माध्यमातून माहिम बेट परेल आणि वरळीशी जोडले गेले होते.']
portuguese_simple = ['Comportamento semelhante tiveram outros mercados de capitais no mundo.', '- O CPC está abaixo do que queremos, apesar do aumento quando comparado com janeiro.', 'As coisas vão voltar à normalidade.', 'O presidente foi recebido por uma platéia reunida por PT, PC do B e PSB , partidos da base do governo.', '- Havia um Fiat Doblò estacionado em frente a uma panificadora.']


for sentence in portuguese_simple:
    for ratio in [0.3, 0.5, 0.8]:
        print(f"Ratio: {ratio*100}%")
        print(code_switch(sentence, ratio))
        print("-----------------------------")

# for sentence in marathi_sentences:
#     translation = get_translation(sentence)
#     print(translation)
