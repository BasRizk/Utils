# -*- coding: utf-8 -*-
dataset_name = "librispeech"
import os

with open("spk_test.txt", "a") as tests_f:
    for folder in os.listdir("./"):
        tests_f.write(dataset_name  + folder + "\n")