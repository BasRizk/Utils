# -*- coding: utf-8 -*-

#import numpy as np
import pandas as pd
import re
#import sys
#
#if not sys.argv[1]:
#    print("Please enter file path.")
#    

log_filename = 'log_session_v9_10_1.log'
old_lines = pd.read_csv(log_filename, sep="\t", header = None, engine='python', error_bad_lines=False)
new_lines = []

caught_load_begin = False
for line in old_lines[0]:
    if caught_load_begin:
        if re.match(r'.*100%', line):
            caught_load_begin = False
            new_lines.append(line)
#            print(line)
        continue
    
    elif re.match(r'[^0-9]*\d\d?%', line):
        caught_load_begin = True
        print(line)
        
    else:
        new_lines.append(line)
        

with open("cleaned_" + log_filename, 'w') as f:
    for line in new_lines:
        f.write("%s\n" % line)