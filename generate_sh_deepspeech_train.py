#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import os.path as path
# =============================================================================
# Methods Definitions
# =============================================================================
def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.rstrip())
        else:
            break
    rc = process.poll()
    return rc

def prepare_dirs(dirs_list, glob_dir, model_lang, version):
    postfix = "_" + str(version)
    version_path = path.join(glob_dir, model_lang + postfix)
    if not path.exists(version_path):
        os.mkdir(version_path)
    else:
        print("Version directory already exists")
    new_dirs_path_list = []
    for one_dir in dirs_list:
        new_dir_path = path.join(version_path, one_dir + postfix)
        if not path.exists(new_dir_path):
            os.mkdir(new_dir_path)
        new_dirs_path_list.append(new_dir_path)
        assert(path.exists(new_dir_path))
    return new_dirs_path_list

# =============================================================================
# Dataset Parameters
# =============================================================================
glob_dir = "/mnt/datasets/"
sec_glob_dir = "/home/ironbas3/Past_Models/"
model_lang = "de"
model_dir = "de_model"
summary_dir = "de_summ"
checkpoint_dir = "de_cp"
lm_dir = "de_lm"
alphabet_config_path = path.join(glob_dir, lm_dir +"/alphabet.txt") 
train_files_path = path.join(glob_dir,"de/clips/train.csv")
dev_files_path = path.join(glob_dir,"de/clips/dev.csv")
test_files_path = path.join(glob_dir,"de/clips/test.csv")
lm_binary_path = path.join(glob_dir, lm_dir + "/de_lm.binary")
lm_trie_path = path.join(glob_dir, lm_dir + "/de_trie")
checkpoint_dir_path = path.join(glob_dir, checkpoint_dir) 
export_dir_path = path.join(glob_dir, model_dir)
summary_dir_path = path.join(glob_dir, summary_dir)
assert(path.exists(alphabet_config_path))
assert(path.exists(train_files_path))
assert(path.exists(dev_files_path))
assert(path.exists(test_files_path))
assert(path.exists(lm_binary_path))
assert(path.exists(lm_trie_path))
assert(path.exists(checkpoint_dir_path))
assert(path.exists(export_dir_path))
assert(path.exists(summary_dir_path))

log_filepath = path.join(glob_dir, "de_training_meta_log.txt")


# =============================================================================
# DNN PARAMETERS
# =============================================================================
train_batch_size = 6
dev_batch_size = 12
test_batch_size = 12
epoch = 30
learning_rate = 0.0001
display_step = 0
validation_step = 1 
dropout_rate = 0.5
n_hidden = 2048
lm_alpha = 0.75
lm_beta = 1.85
dropout_rate = 0.05
####### START NOT USED
beam_width = 1024 
epsilon = 1e-08
beta1 = 0.9
beta2 = 0.999
relu_clip = 20.0
####### END NOT USED

# =============================================================================
# Model meta parameters
# =============================================================================
early_stop = True
checkpoint_step = 1
export_version = 4
if early_stop:
    early_stop_stat = "--early_stop"
else:
    early_stop_stat = "--noearly_stop"
####### START NOT USED
earlystop_nsteps = 4
estop_mean_thresh = 0.5
estop_std_thresh = 0.5
summary_secs = 20 # Every 20 seconds
####### END NOT USED

#tmp_export_dir_path, tmp_summary_dir_path, tmp_checkpoint_dir_path =\
#    export_dir_path, summary_dir_path, checkpoint_dir_path
def train_tune(list_of_parameters, list_of_version_num):
    print("TODO")

def log_training_command(training_command, log_filepath):
    log_file = open(log_filepath, "a+")
    log_file.write(str(training_command) + ",\n")
    log_file.close()

# =============================================================================
# INCREASING BATCH SIZE FOR VERSION 3 ---- TODO NOTE LATER IN FOLDERS
# =============================================================================
train_batch_size = train_batch_size*2
dev_batch_size = dev_batch_size*2
test_batch_size = test_batch_size*2
for dropout_rate in [0.2, 0.3, 0.4]:
    for n_hidden in [1024]:
        version_num = "n_hidden_" + str(n_hidden) + "_dp_" + str(dropout_rate)
        export_dir_path, summary_dir_path, checkpoint_dir_path =\
            prepare_dirs([model_dir,
                          summary_dir,
                          checkpoint_dir], sec_glob_dir, model_lang, version_num)
        training_command = [
                'python', '-u', '/home/ironbas3/DeepSpeech/DeepSpeech.py',
                '--alphabet_config_path', alphabet_config_path,
                '--train_files', train_files_path,
                '--dev_files', dev_files_path,
                '--test_files', test_files_path,
                '--train_batch_size', str(train_batch_size),
                '--dev_batch_size', str(dev_batch_size),
                '--test_batch_size', str(test_batch_size),
                '--epoch', str(epoch),
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
    
        training_process = run_command(training_command)
        log_training_command(training_command, log_filepath)

#for dropout_rate in [0.2, 0.3, 0.4, 0.5]:
#    for n_hidden in [2048]:
#        version_num = "n_hidden_" + str(n_hidden) + "_dp_" + str(dropout_rate)
#        export_dir_path, summary_dir_path, checkpoint_dir_path =\
#            prepare_dirs([model_dir,
#                          summary_dir,
#                          checkpoint_dir], sec_glob_dir, model_lang, version_num)
#        training_command = [
#                'python', '-u', '/home/ironbas3/DeepSpeech/DeepSpeech.py',
#                '--alphabet_config_path', alphabet_config_path,
#                '--train_files', train_files_path,
#                '--dev_files', dev_files_path,
#                '--test_files', test_files_path,
#                '--train_batch_size', str(train_batch_size),
#                '--dev_batch_size', str(dev_batch_size),
#                '--test_batch_size', str(test_batch_size),
#                '--epoch', str(epoch),
#                '--n_hidden', str(n_hidden),
#                '--learning_rate', str(learning_rate),
#                '--display_step', str(display_step),
#                '--validation_step', str(validation_step),
#                '--dropout_rate', str(dropout_rate),
#                '--checkpoint_step', str(checkpoint_step),
#                '--checkpoint_dir', checkpoint_dir_path,
#                '--export_dir', export_dir_path,
#                '--summary_dir', summary_dir_path,
#                '--noexport_tflite',
#                early_stop_stat,
#                #'--remove_export',
#                '--export_version', str(export_version),
#                '--lm_binary_path', lm_binary_path,
#                '--lm_trie_path', lm_trie_path,
#                '--lm_alpha', str(lm_alpha),
#                '--lm_beta', str(lm_beta)
#                ]
#       
#        print("\n>>> Training with version_num: " + version_num +"\n")
#    
#        training_process = run_command(training_command)
#        log_training_command(training_command, log_filepath)
        
#for dropout_rate in [0.4, 0.5]:
#    for n_hidden in [4096]:
#        version_num = "n_hidden_" + str(n_hidden) + "_dp_" + str(dropout_rate)
#        export_dir_path, summary_dir_path, checkpoint_dir_path =\
#            prepare_dirs([model_dir,
#                          summary_dir,
#                          checkpoint_dir], sec_glob_dir, model_lang, version_num)
#        training_command = [
#                'python', '-u', '/home/ironbas3/DeepSpeech/DeepSpeech.py',
#                '--alphabet_config_path', alphabet_config_path,
#                '--train_files', train_files_path,
#                '--dev_files', dev_files_path,
#                '--test_files', test_files_path,
#                '--train_batch_size', str(train_batch_size),
#                '--dev_batch_size', str(dev_batch_size),
#                '--test_batch_size', str(test_batch_size),
#                '--epoch', str(epoch),
#                '--n_hidden', str(n_hidden),
#                '--learning_rate', str(learning_rate),
#                '--display_step', str(display_step),
#                '--validation_step', str(validation_step),
#                '--dropout_rate', str(dropout_rate),
#                '--checkpoint_step', str(checkpoint_step),
#                '--checkpoint_dir', checkpoint_dir_path,
#                '--export_dir', export_dir_path,
#                '--summary_dir', summary_dir_path,
#                '--noexport_tflite',
#                early_stop_stat,
#                #'--remove_export',
#                '--export_version', str(export_version),
#                '--lm_binary_path', lm_binary_path,
#                '--lm_trie_path', lm_trie_path,
#                '--lm_alpha', str(lm_alpha),
#                '--lm_beta', str(lm_beta)
#                ]
#       
#        print("\n>>> Training with version_num: " + version_num +"\n")
#    
#        training_process = run_command(training_command)
#        log_training_command(training_command, log_filepath)

#for dropout_rate in [0.2, 0.3]:
#    for n_hidden in [1024, 2048, 4096]:
#        for learning_rate in [0.01, 0.001, 0.00001]:
#            version_num = "lr_" + str(learning_rate) + "_n_hidden_" + str(n_hidden) + "_dp_" + str(dropout_rate)
#            export_dir_path, summary_dir_path, checkpoint_dir_path =\
#                prepare_dirs([model_dir,
#                              summary_dir,
#                              checkpoint_dir], sec_glob_dir, model_lang, version_num)
#            training_command = [
#                    'python', '-u', '/home/ironbas3/DeepSpeech/DeepSpeech.py',
#                    '--alphabet_config_path', alphabet_config_path,
#                    '--train_files', train_files_path,
#                    '--dev_files', dev_files_path,
#                    '--test_files', test_files_path,
#                    '--train_batch_size', str(train_batch_size),
#                    '--dev_batch_size', str(dev_batch_size),
#                    '--test_batch_size', str(test_batch_size),
#                    '--epoch', str(epoch),
#                    '--n_hidden', str(n_hidden),
#                    '--learning_rate', str(learning_rate),
#                    '--display_step', str(display_step),
#                    '--validation_step', str(validation_step),
#                    '--dropout_rate', str(dropout_rate),
#                    '--checkpoint_step', str(checkpoint_step),
#                    '--checkpoint_dir', checkpoint_dir_path,
#                    '--export_dir', export_dir_path,
#                    '--summary_dir', summary_dir_path,
#                    '--noexport_tflite',
#                    early_stop_stat,
#                    #'--remove_export',
#                    '--export_version', str(export_version),
#                    '--lm_binary_path', lm_binary_path,
#                    '--lm_trie_path', lm_trie_path,
#                    '--lm_alpha', str(lm_alpha),
#                    '--lm_beta', str(lm_beta)
#                    ]
#           
#            print("\n>>> Training with version_num: " + version_num +"\n")
#        
#            training_process = run_command(training_command)
#            log_training_command(training_command, log_filepath)

for dropout_rate in [0.2, 0.3, 0.4]:
    for n_hidden in [1024, 2048, 4096]:
        for train_batch_size in [3, 12, 24]:
            dev_batch_size = train_batch_size*2
            test_batch_size = train_batch_size*2
            version_num = "tr_batch_" + str(train_batch_size) + "_n_hidden_" + str(n_hidden) + "_dp_" + str(dropout_rate)
            export_dir_path, summary_dir_path, checkpoint_dir_path =\
                prepare_dirs([model_dir,
                              summary_dir,
                              checkpoint_dir], sec_glob_dir, model_lang, version_num)
            training_command = [
                    'python', '-u', '/home/ironbas3/DeepSpeech/DeepSpeech.py',
                    '--alphabet_config_path', alphabet_config_path,
                    '--train_files', train_files_path,
                    '--dev_files', dev_files_path,
                    '--test_files', test_files_path,
                    '--train_batch_size', str(train_batch_size),
                    '--dev_batch_size', str(dev_batch_size),
                    '--test_batch_size', str(test_batch_size),
                    '--epoch', str(epoch),
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
        
            training_process = run_command(training_command)
            log_training_command(training_command, log_filepath)
            
        
print("..DONE..")
    












