# -*- coding: utf-8 -*-

# =============================================================================
#  Normalizing text for Kenlm to create Language model
# =============================================================================
import unicodedata
import pandas as pd
from text import Alphabet, validate_label

isText = False

alphabet_path = "/mnt/datasets/de_lm/alphabet.txt"
alphabet = Alphabet(alphabet_path)

if isText:
    data = pd.read_csv('german_text_for_model.txt', sep="\t", header = None)
else:    
    filename = "test.csv"
    data = pd.read_csv('/home/ironbas3/SpeechDS/GERMAN/data/' + filename)

def label_filter(label):
    
    try:
        label = unicodedata.normalize("NFKD", label.strip()) \
            .encode("ascii", "ignore") \
            .decode("ascii", "ignore") \
            .replace('\d+', '')
        label = validate_label(label)
        if alphabet and label:
            try:
                [alphabet.label_from_string(c) for c in label]
            except KeyError:
                label = None
    except:
        print(label)
            
    return label
    
    

if isText:
    data.applymap(label_filter)
    data.to_csv(r'german_text_for_model_normalized.txt', header=None, index=None, sep=' ', mode='a')
else:
    data = data.dropna()
    data[data["transcript"].str.find(r'\d+') == -1]
    data["transcript"] = data["transcript"].apply(label_filter)
#    for i in range(len(data["transcript"])):
#        data["transcript"][i] = label_filter(data["transcript"][i])
    data.to_csv("normalized_texts/" + filename, sep=',', mode='a', index=False)
