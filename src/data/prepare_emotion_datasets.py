import os
import json
from datasets import load_dataset
from collections import Counter
import pandas as pd
import requests, zipfile, io
import glob
import csv

# Unified label set (GoEmotions 28 labels)
GOEMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise'
]
LABEL_MAP = {label: i for i, label in enumerate(GOEMOTIONS_LABELS)}

os.makedirs('data/processed/combined', exist_ok=True)

# Helper to save as JSONL
def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def print_stats(name, data):
    print(f"{name}: {len(data)} samples")
    all_labels = [l for d in data for l in d['labels']]
    print(Counter(all_labels))

# 1. GoEmotions
print("Loading GoEmotions...")
ge = load_dataset('go_emotions', 'simplified')
train = [{'text': t, 'labels': l} for t, l in zip(ge['train']['text'], ge['train']['labels'])]
val = [{'text': t, 'labels': l} for t, l in zip(ge['validation']['text'], ge['validation']['labels'])]
test = [{'text': t, 'labels': l} for t, l in zip(ge['test']['text'], ge['test']['labels'])]
print_stats('GoEmotions train', train)
save_jsonl(train, 'data/processed/combined/goemotions_train.jsonl')
save_jsonl(val, 'data/processed/combined/goemotions_val.jsonl')
save_jsonl(test, 'data/processed/combined/goemotions_test.jsonl')

# 2. DailyDialog
print("Loading DailyDialog...")
dd = load_dataset('daily_dialog', trust_remote_code=True)
dd_map = {
    0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'sadness', 5: 'surprise', 6: 'neutral'
}
dd_data = []
for split in ['train', 'validation', 'test']:
    for text, labels in zip(dd[split]['dialog'], dd[split]['emotion']):
        for utt, lab in zip(text, labels):
            mapped = [LABEL_MAP[dd_map.get(lab, 'neutral')]] if lab != -1 else [LABEL_MAP['neutral']]
            dd_data.append({'text': utt, 'labels': mapped})
print_stats('DailyDialog', dd_data)
save_jsonl(dd_data, 'data/processed/combined/dailydialog.jsonl')

# 3. EmotionLines (from GitHub)
# print("Loading EmotionLines from GitHub...")
# EMOTIONLINES_URL = "https://github.com/emorynlp/EmotionLines/archive/refs/heads/master.zip"
# local_zip = "data/processed/combined/emotionlines.zip"
# local_dir = "data/processed/combined/emotionlines_friends/"
# if not os.path.exists(local_dir):
#     print("Downloading EmotionLines...")
#     r = requests.get(EMOTIONLINES_URL)
#     with open(local_zip, 'wb') as f:
#         f.write(r.content)
#     with zipfile.ZipFile(local_zip, 'r') as zip_ref:
#         zip_ref.extractall(local_dir)
#
# # Parse Friends split
# el_map = {
#     'neutral': 'neutral', 'joy': 'joy', 'sadness': 'sadness', 'fear': 'fear',
#     'anger': 'anger', 'surprise': 'surprise', 'disgust': 'disgust', 'non-neutral': 'neutral'
# }
# el_data = []
# for split in ['train', 'dev', 'test']:
#     split_path = glob.glob(f"{local_dir}/EmotionLines-master/dataset/friends/{split}/*.json")
#     for file in split_path:
#         with open(file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 d = json.loads(line)
#                 utt = d['utterance']
#                 lab = d['emotion']
#                 mapped = [LABEL_MAP[el_map.get(lab, 'neutral')]]
#                 el_data.append({'text': utt, 'labels': mapped})
# print_stats('EmotionLines', el_data)
# save_jsonl(el_data, 'data/processed/combined/emotionlines.jsonl')

# 4. EmpatheticDialogues
# print("Loading EmpatheticDialogues...")
# ed = load_dataset('empathetic_dialogues', trust_remote_code=True)
# ed_map = {l: l if l in LABEL_MAP else 'neutral' for l in set(ed['train']['emotion'])}
# ed_data = []
# for split in ['train', 'validation', 'test']:
#     for utt, lab in zip(ed[split]['utterance'], ed[split]['emotion']):
#         mapped = [LABEL_MAP[ed_map.get(lab, 'neutral')]]
#         ed_data.append({'text': utt, 'labels': mapped})
# print_stats('EmpatheticDialogues', ed_data)
# save_jsonl(ed_data, 'data/processed/combined/empatheticdialogues.jsonl')

# 5. MELD (text only)
print("Loading MELD...")
meld_map = {
    'neutral': 'neutral', 'joy': 'joy', 'sadness': 'sadness', 'fear': 'fear',
    'anger': 'anger', 'surprise': 'surprise', 'disgust': 'disgust'
}
def parse_meld_csv(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            utt = row['Utterance']
            lab = row['Emotion'].strip().lower()
            mapped = [LABEL_MAP[meld_map.get(lab, 'neutral')]]
            data.append({'text': utt, 'labels': mapped})
    return data
meld_train = parse_meld_csv('data/raw/meld/MELD.Raw/train_sent_emo.csv')
meld_val = parse_meld_csv('data/raw/meld/MELD.Raw/dev_sent_emo.csv')
meld_test = parse_meld_csv('data/raw/meld/MELD.Raw/test_sent_emo.csv')
print_stats('MELD train', meld_train)
save_jsonl(meld_train, 'data/processed/combined/meld_train.jsonl')
print_stats('MELD val', meld_val)
save_jsonl(meld_val, 'data/processed/combined/meld_val.jsonl')
print_stats('MELD test', meld_test)
save_jsonl(meld_test, 'data/processed/combined/meld_test.jsonl')

print("All datasets processed and saved in data/processed/combined/") 