# -*- coding: utf-8 -*-
import pandas as pd
import os

toChangeStr = "^"
toBeStr = ""
num_edited_lines = 0
def label_filter(label):
    global toChangeStr
    global toBeStr
    global num_edited_lines

    new_label = label.replace(toChangeStr, toBeStr)
    if new_label != label:
        num_edited_lines += 1
        if num_edited_lines < 10:
            print(new_label)
    return new_label

for a_dir in os.listdir():
    a_dir
    for filename in ["train", "test", "dev"]:
        csvfile =  os.path.join(a_dir, str(filename) + ".csv")
        ds = pd.read_csv(csvfile, sep=",")
        
        num_edited_lines = 0
        ds["transcript"] = ds["transcript"].map(label_filter)
        print("File: " + csvfile + " Edited lines = " + str(num_edited_lines))
            
        edited_filename = csvfile +"_edited.csv"
        ds.to_csv(edited_filename, sep=",", index = None)
        
        os.rename(csvfile, csvfile + "_oldtranscription.csv")
        os.rename(edited_filename, csvfile)

