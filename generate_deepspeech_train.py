#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import os.path as path
import time

# =============================================================================
# Methods Definitions
# =============================================================================
def run_command(command, polling = True):
    if not polling:
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        return
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

def open_log_session():
    return
#    localtime = time.strftime("%Y%m%d-%H%M%S")
#    log_filename = "log_deepspeech_train_session_" + str(localtime) + ".log"
#    log_command = ["script", log_filename]
#    run_command(log_command, polling=False)
#    print("\n Begin logging at" + log_filename + "\n")
#    return log_filename

def close_log_session(log_filename):
    return
#    log_command = ["exit"]
#    run_command(log_command, polling=False)
#    print("\n Writing log " + log_filename + "\n")

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

num_of_trainings = 0

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
export_version = 5
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
# Make it run more
# =============================================================================
print("\n>>>>>>>>>>>>>>>>.. TO BE RUNNING MORE >>>>>>>>>>>>>>>>>>>>\n" )
print(">>>>>>>>>>>>>>>>.. TO BE RUNNING MORE >>>>>>>>>>>>>>>>>>>>\n")
print(">>>>>>>>>>>>>>>>.. TO BE RUNNING MORE >>>>>>>>>>>>>>>>>>>>\n" )

for dropout_rate in [0.4]:
    for n_hidden in [1024]:
        
        log_session = open_log_session()
        num_of_trainings += 1
        print("\n\nTraining number " + str(num_of_trainings) + "\n\n")
        
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
        close_log_session(log_session)

for dropout_rate in [0.5]:
    for n_hidden in [2048, 4096]:
        
        log_session = open_log_session()
        num_of_trainings += 1
        print("\n\nTraining number " + str(num_of_trainings) + "\n\n")
        
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
    
        close_log_session(log_session)
        training_process = run_command(training_command)
        log_training_command(training_command, log_filepath)
        
        
# =============================================================================
# INCREASING BATCH SIZE FOR VERSION 3 ---- TODO NOTE LATER IN FOLDERS
# =============================================================================
        
print("\n>>>>>>>>>>>>>>>>.. INCREASING BATCH SIZE >>>>>>>>>>>>>>>>>>>>\n" )
print(">>>>>>>>>>>>>>>>.. INCREASING BATCH SIZE >>>>>>>>>>>>>>>>>>>>\n")
print(">>>>>>>>>>>>>>>>.. INCREASING BATCH SIZE >>>>>>>>>>>>>>>>>>>>\n" )

_train_batch_size = train_batch_size*2
_dev_batch_size = dev_batch_size*2
_test_batch_size = test_batch_size*2
for dropout_rate in [0.2, 0.3]:
    for n_hidden in [1024, 2048]:
        
        log_session = open_log_session()
        num_of_trainings += 1
        print("\n\nTraining number " + str(num_of_trainings) + "\n\n")
        
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
                '--train_batch_size', str(_train_batch_size),
                '--dev_batch_size', str(_dev_batch_size),
                '--test_batch_size', str(_test_batch_size),
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
    
        close_log_session(log_session)
        training_process = run_command(training_command)
        log_training_command(training_command, log_filepath)

for dropout_rate in [0.4]:
    for n_hidden in [2048, 4096]:
        
        log_session = open_log_session()
        num_of_trainings += 1
        print("\n\nTraining number " + str(num_of_trainings) + "\n\n")
        
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
                '--train_batch_size', str(_train_batch_size),
                '--dev_batch_size', str(_dev_batch_size),
                '--test_batch_size', str(_test_batch_size),
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
    
        close_log_session(log_session)
        training_process = run_command(training_command)
        log_training_command(training_command, log_filepath)
        
        
# =============================================================================
# INCREASING NUMBER OF EPOCHS
# =============================================================================
print("\n>>>>>>>>>>>>>>>>.. INCREASING NUMBER OF EPOCHS >>>>>>>>>>>>>>>>>>>>\n" )
print(">>>>>>>>>>>>>>>>.. INCREASING NUMBER OF EPOCHS >>>>>>>>>>>>>>>>>>>>\n")
print(">>>>>>>>>>>>>>>>.. INCREASING NUMBER OF EPOCHS >>>>>>>>>>>>>>>>>>>>\n" )

for dropout_rate in [0.2]:
    for n_hidden in [1024, 2048]:
        for learning_rate in [0.00001]:
            
            log_session = open_log_session()
            num_of_trainings += 1
            print("\n\nTraining number " + str(num_of_trainings) + "\n\n")
            
            version_num = "lr_" + str(learning_rate) + "_n_hidden_" + str(n_hidden) + "_dp_" + str(dropout_rate)
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
                    '--epoch', str(-20),
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

# =============================================================================
# RUN FOR FIRST TIME 
# =============================================================================
print("\n>>>>>>>>>>>>>>>>.. RUN FOR FIRST TIME  >>>>>>>>>>>>>>>>>>>>\n" )
print(">>>>>>>>>>>>>>>>.. RUN FOR FIRST TIME  >>>>>>>>>>>>>>>>>>>>\n")
print(">>>>>>>>>>>>>>>>.. RUN FOR FIRST TIME  >>>>>>>>>>>>>>>>>>>>\n" )
for dropout_rate in [0.2]:
    for n_hidden in [4096]:
        for learning_rate in [0.01, 0.001, 0.00001]:
            
            log_session = open_log_session()
            num_of_trainings += 1
            print("\n\nTraining number " + str(num_of_trainings) + "\n\n")
            
            version_num = "lr_" + str(learning_rate) + "_n_hidden_" + str(n_hidden) + "_dp_" + str(dropout_rate)
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
        
            close_log_session(log_session)
            training_process = run_command(training_command)
            log_training_command(training_command, log_filepath)

for dropout_rate in [0.3]:
    for n_hidden in [1024, 2048, 4096]:
        for learning_rate in [0.01, 0.001, 0.00001]:
            
            log_session = open_log_session()
            num_of_trainings += 1
            print("\n\nTraining number " + str(num_of_trainings) + "\n\n")
            
            version_num = "lr_" + str(learning_rate) + "_n_hidden_" + str(n_hidden) + "_dp_" + str(dropout_rate)
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
        
            close_log_session(log_session)
            training_process = run_command(training_command)
            log_training_command(training_command, log_filepath)

# =============================================================================
# TRYING DIFFERENT BATCH SIZES FROM BEGINNING
# =============================================================================
print("\n>>>>>>>>>>>>>>>>.. TRYING DIFFERENT BATCH SIZES FROM BEGINNING  >>>>>>>>>>>>>>>>>>>>\n" )
print(">>>>>>>>>>>>>>>>.. TRYING DIFFERENT BATCH SIZES FROM BEGINNING  >>>>>>>>>>>>>>>>>>>>\n")
print(">>>>>>>>>>>>>>>>.. TRYING DIFFERENT BATCH SIZES FROM BEGINNING  >>>>>>>>>>>>>>>>>>>>\n" )

learning_rate = 0.0001
for dropout_rate in [0.2, 0.3, 0.4]:
    for n_hidden in [1024, 2048, 4096]:
        for learning_rate in [0.0001, 0.00001]:
            for train_batch_size in [3, 12, 24]:
        
                log_session = open_log_session()
                num_of_trainings += 1
                print("\n\nTraining number " + str(num_of_trainings) + "\n\n")
                
                dev_batch_size = train_batch_size*2
                test_batch_size = train_batch_size*2
                version_num = "tr_batch_" + str(train_batch_size) + "_lr_" + str(learning_rate) + + "_n_hidden_" + str(n_hidden) + "_dp_" + str(dropout_rate)
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
            
                close_log_session(log_session)
                training_process = run_command(training_command)
                log_training_command(training_command, log_filepath)
            
        
print("..DONE..")
    












