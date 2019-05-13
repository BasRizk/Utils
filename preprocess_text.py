# -*- coding: utf-8 -*-

# =============================================================================
#  Normalizing text for Kenlm to create Language model
# =============================================================================
import unicodedata
import pandas as pd
from text import Alphabet, validate_label

alphabet_path = "/mnt/datasets/de_model/alphabet.txt"

data = pd.read_csv('german_text_for_model.txt', sep="\t", header = None)

alphabet = Alphabet(alphabet_path)

def label_filter(label):
        label = unicodedata.normalize("NFKD", label.strip()) \
            .encode("ascii", "ignore") \
            .decode("ascii", "ignore")
        label = validate_label(label)
        if alphabet and label:
            try:
                [alphabet.label_from_string(c) for c in label]
            except KeyError:
                label = None
        return label
    
    
data.applymap(label_filter)

data.to_csv(r'german_text_for_model_normalized.txt', header=None, index=None, sep=' ', mode='a')