import torch
import itertools
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
import random
import os
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # SET the GPUs you want to use


class TextAligner:
    def __init__(self):
        self.model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co", )
        self.tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")
        self.align_layer = 8
        self.threshold = 1e-3

    def align_texts(self, original, translated):
        sent_src, sent_tgt = original.strip().split(), translated.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=self.tokenizer.model_max_length, truncation=True)['input_ids'], self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=self.tokenizer.model_max_length)['input_ids']

        sub2word_map_src = [i for i, word_list in enumerate(token_src) for _ in word_list]
        sub2word_map_tgt = [i for i, word_list in enumerate(token_tgt) for _ in word_list]

        self.model.eval()
        with torch.no_grad():
            out_src = self.model(ids_src.unsqueeze(0), output_hidden_states=True)[2][self.align_layer][0, 1:-1]
            out_tgt = self.model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][self.align_layer][0, 1:-1]

            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_inter = (softmax_srctgt > self.threshold)*(softmax_tgtsrc > self.threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = {(sent_src[sub2word_map_src[i]], sent_tgt[sub2word_map_tgt[j]]) for i, j in align_subwords}

        return align_words

# Load the NLLB model for translation
#tel_Telu
#hin_Deva
#mar_Deva
#ben_Beng
#vie_Latn
#ces_Latn
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="hin_Deva")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

# def get_translation(word):
#     """Fetch the English translation for a given Telugu word using the NLLB model."""
#     inputs = tokenizer(word, return_tensors="pt")
#     translated_tokens = model.generate(
#         **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=100
#     )
#     english_phrase = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
#     return english_phrase

class CodeSwitcher(TextAligner):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")

    def switch_content_words(self, source_sentences, ratio=0.5):
        english_translations = ['Uhh tell me, is this not a service?', 'But getting stuck here would not help.', 'Bajpai was also included among these economists.', 'I thought, um...should I go see a movie this evening?', 'He...means, he bought his new car.']
        mixed_sentences = []

        for source, english in zip(source_sentences, english_translations):
            aligned_pairs = self.align_texts(source, english)
            print(aligned_pairs)
            aligned_dict = dict(aligned_pairs)
            print(aligned_dict)

            doc = self.nlp(english)
            content_word_tags = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "RB", "RBR", "RBS", "NOUN", "VERB", "ADJ", "ADV"]
            content_words = [token.text for token in doc if token.tag_ in content_word_tags]

            num_to_switch = int(len(content_words) * ratio)
            words_to_switch = random.sample(content_words, min(num_to_switch, len(content_words)))

            new_sentence = []
            for word in source.split():
                aligned_english_word = aligned_dict.get(word, None)
                if aligned_english_word and aligned_english_word in words_to_switch:
                    new_sentence.append(aligned_english_word)
                else:
                    new_sentence.append(word)

            mixed_sentences.append(' '.join(new_sentence))

        return mixed_sentences

# Usage:
switcher = CodeSwitcher()
hindi_sentences = ['अ मुझे बताइए ये सेवा नहीं है क्या?', 'लेकिन यहां पर पर अटकने से काम नहीं होगा।', 'बाजपेयी भी इन अर अर्थशास्त्रियों में शामिल थे।', 'मैंने सोचा कि, उम्म...क्या मैं आज शाम को फिल्म देखने जाऊँ?', 'उसके...मतलब, उसने अपनी नई कार खरीदी है।']
# french_sentences = ["Je veux que tu envoies la photo photo d' écran à Mireille avec Lucie en cc.", "Envoie un mail à euh jena@polonium.com.", "envoie une un message à Alice.", "Peux-tu euuh envoyer cet SMS sur le chien de maman?", "Dis à Karen par euh SMS que j'arrive en en joignant mon heure d'arrivée.", "Mets en pièce jointe mes coordonnées GPS au courriel pour euh Lucie.", "écris une un mail à Alice.", "Merci d'écrire un email à Pierre."]
print(switcher.switch_content_words(hindi_sentences, 0.6))
print("-----------------")
print(switcher.switch_content_words(hindi_sentences, 0.8))
print("-----------------")
print(switcher.switch_content_words(hindi_sentences, 1.0))
