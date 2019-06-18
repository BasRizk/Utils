# -*- coding: utf-8 -*-

#import pandas as pd


filename = "BOOKS.TXT"
clean_word = "test-clean"
#seperator = "|"
#my_columns = ['ID','SEX','SUBSET','MINUTES','NAME']
#speakers_ds = pd.read_csv(filename, sep= seperator, skiprows=12, names = my_columns, squeeze=True)
#speakers_ds = speakers_ds[~speakers_ds.SUBSET.str.contains(clean_word, na=False)]
#speakers_ds.to_csv("CLEANED_" + filename, index=False)
#

with open(filename, "r") as f, open("CLEANED_" + filename, "w+") as cleaned:
    for line in f.readlines():
        if line[0].isdigit:
            if clean_word in line:
                continue
        cleaned.write(line)