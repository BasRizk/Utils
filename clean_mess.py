# -*- coding: utf-8 -*-

import pandas as pd
import re

non_alphabet = r"[^A-Z ^a-z ^']+"
num_non_alphabet_lines = 0

def label_filter(label):
    global non_alphabet
    global num_non_alphabet_lines
    new_label = re.sub(non_alphabet, '', label)
    if new_label != label:
        print("Changed label: " + label + " => " + new_label)
        num_non_alphabet_lines += 1    
    return new_label
    
for i in [0, 1, 2]:
    filename = "transcripts_0" + str(i) + ".csv"
    ds = pd.read_csv(filename, sep=";", header =None)
    
    
    ds[3] = ds[3].map(label_filter)
    ds[3] = ds[3].str.upper()
    ds[4] = ds[3].str.lower()
    
    # QUALITY UPDGRADE
    ds[5] = 2
    ds = ds.dropna()
    
    
    ds = ds.drop( ds[ds[3].str.contains(";")].index)
    
    ds.to_csv( filename +"_clean.csv",sep=";", header=None, index = None)
