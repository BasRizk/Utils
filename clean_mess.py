# -*- coding: utf-8 -*-

import pandas as pd

filename = "transcripts_0.csv"
ds = pd.read_csv(filename, sep=";", header =None)
#ds = ds.drop(ds.columns[0], axis=1)
#ds[4] = ds[3].str.lower()

# QUALITY UPDGRADE
ds[5] = 2

ds.to_csv( filename +"_clean.csv",sep=";", header=None, index = None)