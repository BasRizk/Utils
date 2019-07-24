# -*- coding: utf-8 -*-

import pandas as pd
import os

toChangeStr = "/ironbas3/"
toBeStr = "/ironbas0/"
num_edited_lines = 0
def label_filter(label):
    global toChangeStr
    global toBeStr
    global num_edited_lines

    new_label = label.replace(toChangeStr, toBeStr)
    if new_label != label:
        num_edited_lines += 1    
    return new_label

for a_dir in os.listdir():
    a_dir
    for filename in ["train", "test", "dev"]:
        csvfile =  os.path.join(a_dir, str(filename) + ".csv")
        ds = pd.read_csv(csvfile, sep=",")
        
        num_edited_lines = 0
        ds["wav_filename"] = ds["wav_filename"].map(label_filter)
        print("File: " + csvfile + " Edited lines = " + str(num_edited_lines))
        if num_edited_lines == len(ds):
            print( " => perfect.\n")
        else:
            print( " => WARNING :: something is wrong!!\n")
            
        edited_filename = csvfile +"_edited.csv"
        ds.to_csv(edited_filename, sep=",", index = None)
        
        os.rename(csvfile, csvfile + "_old.csv")
        os.rename(edited_filename, csvfile)

