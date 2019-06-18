# -*- coding: utf-8 -*-
import subprocess
import os
import os.path as path

# =============================================================================
# Helper Methods Definitions
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
    # TODO - fix open log sesion without interrupt
    return
#    localtime = time.strftime("%Y%m%d-%H%M%S")
#    log_filename = "log_deepspeech_train_session_" + str(localtime) + ".log"
#    log_command = ["script", log_filename]
#    run_command(log_command, polling=False)
#    print("\n Begin logging at" + log_filename + "\n")
#    env = "deepspeechTrain"
#    env_command = ["conda", "activate", env]
#    run_command(env_command, polling=False)
#    print("\n Env " + env  + " is supposedly activated\n")
#    return log_filename

def close_log_session(log_filename):
    # TODO -  upon open log sesion fix
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

def log_training_command(training_command, log_filepath):
    log_file = open(log_filepath, "a+")
    log_file.write(str(training_command) + ",\n")
    log_file.close()