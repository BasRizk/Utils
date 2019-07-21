# -*- coding: utf-8 -*-

import pandas as pd

def does_it_exist(bag, item, label_to_ignore):
    found = bag.loc[bag[0].str.startswith(item) & bag[0].str.contains(label_to_ignore)].any()[0]
    if found:
        return True
    return False
    

# LIBRISPEECH

spk_test_file = "spk_test.txt"
to_be_test = pd.read_csv(spk_test_file, header=None)
to_be_test[0] = to_be_test[0].str.replace("librispeech", "")
label_to_ignore = "other"

speakers_file = "SPEAKERS.TXT"
speakers_info = pd.read_csv(speakers_file, sep=";", skiprows = 12, header =None)

non_valid_indexes = []
for index, row in to_be_test.iterrows():
    item = row[0] + " "
    print("at item: " + item + " => ")

    if does_it_exist(speakers_info, item, label_to_ignore) :
        print("Should be deleted.\n")
        non_valid_indexes.append(index)
    else:
        print("accepted.'\n")
    
to_be_test_valid = to_be_test.drop(non_valid_indexes)

to_be_test[0] = "librispeech" + to_be_test[0]

to_be_test_valid.to_csv( spk_test_file +"_clean.csv",sep=";", header=None, index = None)

