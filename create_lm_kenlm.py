# -*- coding: utf-8 -*-

import gzip
import io
import os
import subprocess
from urllib import request

## Grab corpus.
#url = 'http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz'
#data_upper = '/tmp/upper.txt.gz'
#request.urlretrieve(url, data_upper)
#
## Convert to lowercase and cleanup.
#data_lower = '/tmp/lower.txt'
#with open(data_lower, 'w', encoding='utf-8') as lower:
#    with io.TextIOWrapper(io.BufferedReader(gzip.open(data_upper)), encoding='utf8') as upper:
#        for line in upper:
#            lower.write(line.lower())
#
## Build pruned LM.
lm_path = '/home/ironbas3/Downloads/Zamia-Speech Models/generic_en_lang_model_large-r20190501.arpa'
#build = subprocess.call([
#                    "lmplz", "--order", "5",
#                       "--temp_prefix", "/tmp/",
#                       "--memory", "50%",
#                       "--text", data_lower,
#                       "--arpa", lm_path,
#                       "--prune", "0", "0", "0", "1"
#                        ])


# Quantize and produce trie binary.
binary_path = '/home/ironbas3/Downloads/Zamia-Speech Models/en_lm.binary'
build = subprocess.call([
                    "build_binary", "-a", "255",
                          "-q", "8",
                          "trie",
                          lm_path,
                          binary_path
                          ])
#os.remove(lm_path)