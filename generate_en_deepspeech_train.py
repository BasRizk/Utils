#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as path
from utils import run_command, open_log_session, close_log_session
from utils import prepare_dirs, log_training_command
#import time

DEEPSPEECH_VERSION="v0.5.1"
# =============================================================================
# Dataset Parameters
# =============================================================================
glob_dir = "/home/ironbas3/zamia-speech/data/dst/asr-models/deepspeech/librispeech-full"
sec_glob_dir = "/home/ironbas3/zamia-speech/data/dst/asr-models/deepspeech/librispeech-full/models"
model_lang = "en"
model_dir = "en_model"
summary_dir = "en_summ"
checkpoint_dir = "en_cp"
lm_dir = "data/"

alphabet_config_path = path.join(glob_dir, lm_dir +"/alphabet.txt") 

data_dirs = [glob_dir, "/home/ironbas3/SpeechDS"]
train_files_pathes = [path.join(data_dirs[0],"train.csv")]

dev_files_path = path.join(glob_dir,"dev.csv")
test_files_path = path.join(glob_dir,"test.csv")
lm_binary_path = path.join(glob_dir, lm_dir + "/en_lm.binary")
lm_trie_path = path.join(glob_dir, lm_dir + "/trie")
checkpoint_dir_path = path.join(glob_dir, checkpoint_dir) 
export_dir_path = path.join(glob_dir, model_dir)
summary_dir_path = path.join(glob_dir, summary_dir)
assert(path.exists(alphabet_config_path))

train_files_path = ""
for file_path in train_files_pathes:
    assert(path.exists(file_path))
    train_files_path += file_path + ","
train_files_path = train_files_path[:-1]

assert(path.exists(dev_files_path))
assert(path.exists(test_files_path))
assert(path.exists(lm_binary_path))
assert(path.exists(lm_trie_path))
#train_files_path = train_files_path_1 + "," + train_files_path_2

log_filepath = path.join(glob_dir, "en_training_meta_log.txt")

num_of_trainings = 0

def train_tune(drop_outs, n_hiddens, learning_rates, train_batch_sizes, epochs = 30, early_stop = True, export_version = 1):
    # DNN PARAMETERS
    display_step = 0
    validation_step = 1 
    lm_alpha = 0.75
    lm_beta = 1.85
    ####### START NOT USED
#    beam_width = 1024 
#    epsilon = 1e-08
#    beta1 = 0.9
#    beta2 = 0.999
#    relu_clip = 20.0
    ####### END NOT USED
    
    # Model meta parameters
    checkpoint_step = 1
    if early_stop:
        early_stop_stat = "--early_stop"
    else:
        early_stop_stat = "--noearly_stop"
    ####### START NOT USED
#    earlystop_nsteps = 4
#    estop_mean_thresh = 0.5
#    estop_std_thresh = 0.5
#    summary_secs = 20 # Every 20 seconds
    ####### END NOT USED 
    
    # Training loop
    num_of_trainings = 0 
    for dropout_rate in drop_outs:
        for n_hidden in n_hiddens:
            for learning_rate in learning_rates:
                for train_batch_size in train_batch_sizes:
            
                    log_session = open_log_session()
                    num_of_trainings += 1
                    print("\n\nTraining number " + str(num_of_trainings) + "\n\n")
                    
                    dev_batch_size = train_batch_size*2
                    test_batch_size = train_batch_size*2
                    print("Batch sizes (Train, Dev, Test) = %d, %d, %d", train_batch_size, dev_batch_size, test_batch_size)
                    version_num = "tr_batch_" + str(train_batch_size) + "_lr_" + str(learning_rate) + "_n_hidden_" + str(n_hidden) + "_dp_" + str(dropout_rate)
                    export_dir_path, summary_dir_path, checkpoint_dir_path =\
                        prepare_dirs([model_dir,
                                      summary_dir,
                                      checkpoint_dir], sec_glob_dir, model_lang, version_num)
    
                    assert(path.exists(export_dir_path))
                    assert(path.exists(summary_dir_path))
                    assert(path.exists(checkpoint_dir_path))

                    training_command = [
                            'python', '-u', '/home/ironbas3/DeepSpeech/DeepSpeech.py',
                            '--alphabet_config_path', alphabet_config_path,
                            '--train_files', train_files_path,
                            '--dev_files', dev_files_path,
                            '--test_files', test_files_path,
                            '--train_batch_size', str(train_batch_size),
                            '--dev_batch_size', str(dev_batch_size),
                            '--test_batch_size', str(test_batch_size),
                            '--epochs', str(epochs),
                            '--n_hidden', str(n_hidden),
                            '--learning_rate', str(learning_rate),
                            '--display_step', str(display_step),
                            '--validation_step', str(validation_step),
                            '--dropout_rate', str(dropout_rate),
                            '--checkpoint_step', str(checkpoint_step),
                            '--checkpoint_dir', checkpoint_dir_path,
                            '--export_dir', export_dir_path,
                            '--summary_dir', summary_dir_path,
                            '--noexport_tflite',
                            early_stop_stat,
                            #'--remove_export',
                            '--export_version', str(export_version),
                            '--lm_binary_path', lm_binary_path,
                            '--lm_trie_path', lm_trie_path,
                            '--lm_alpha', str(lm_alpha),
                            '--lm_beta', str(lm_beta)
                            ]
                   
                    print("\n>>> Training with version_num: " + version_num +"\n")
                    close_log_session(log_session)
                    training_process = run_command(training_command)
                    log_training_command(training_command, log_filepath)

    return num_of_trainings, training_process
         
## =============================================================================

    
# V1 :: 
train_batch_sizes = [24]
n_hiddens = [2048]
learning_rates = [0.0001]
dropouts = [0.15]
num_of_just_trainings, _ = train_tune(dropouts, n_hiddens, learning_rates, train_batch_sizes, export_version = 10)
num_of_trainings += num_of_just_trainings







