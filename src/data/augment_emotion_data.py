import json
import random
from tqdm import tqdm
from transformers import pipeline, MarianMTModel, MarianTokenizer
import nltk
from nltk.corpus import wordnet
import os

nltk.download('wordnet')

# Paraphrasing pipeline (T5-base)
paraphraser = pipeline('text2text-generation', model='Vamsi/T5_Paraphrase_Paws')

# Back-translation pipeline (English -> French -> English)
fr_model_name = 'Helsinki-NLP/opus-mt-en-fr'
fr_model = MarianMTModel.from_pretrained(fr_model_name)
fr_tokenizer = MarianTokenizer.from_pretrained(fr_model_name)
en_model_name = 'Helsinki-NLP/opus-mt-fr-en'
en_model = MarianMTModel.from_pretrained(en_model_name)
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)

def back_translate(text):
    fr = fr_model.generate(**fr_tokenizer(text, return_tensors="pt", padding=True))
    fr_text = fr_tokenizer.batch_decode(fr, skip_special_tokens=True)[0]
    en = en_model.generate(**en_tokenizer(fr_text, return_tensors="pt", padding=True))
    return en_tokenizer.batch_decode(en, skip_special_tokens=True)[0]

def synonym_replace(text):
    words = text.split()
    new_words = []
    for w in words:
        syns = wordnet.synsets(w)
        lemmas = set(l.name() for s in syns for l in s.lemmas() if l.name() != w)
        if lemmas:
            new_words.append(random.choice(list(lemmas)))
        else:
            new_words.append(w)
    return ' '.join(new_words)

def augment_file(input_path, output_path, n_aug=1):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    augmented = []
    for item in tqdm(data, desc="Augmenting"):
        for _ in range(n_aug):
            # Paraphrase
            para = paraphraser(item['text'], max_length=60, num_return_sequences=1)[0]['generated_text']
            # Back-translate
            bt = back_translate(item['text'])
            # Synonym replace
            syn = synonym_replace(item['text'])
            for aug_text in [para, bt, syn]:
                augmented.append({'text': aug_text, 'labels': item['labels']})
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in augmented:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Augmented {len(data)} samples to {len(augmented)} samples.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n_aug', type=int, default=1)
    args = parser.parse_args()
    augment_file(args.input, args.output, args.n_aug) 