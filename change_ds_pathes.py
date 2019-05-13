# -*- coding: utf-8 -*-
import pandas as pd
import soundfile as sf

files_to_reconfigure = ['train.tsv', 'dev.tsv', 'test.tsv']
##############################################################################
# ---Files Configs
##############################################################################
def convert_then_write_as_wav(audio_path, sample_rate = 16000):
    audio_path = audio_path.split(".")[0]
    audio, fs = sf.read(audio_path + ".mp3")
    sf.write(audio_path + ".wav", audio, sample_rate)
    audio_len = len(audio)/fs 
    return audio_len  

##############################################################################
# ---Pathes Configs
##############################################################################

for tsv_file in files_to_reconfigure:
    path_prefix="/home/ironbas3/SpeechDS/de/clips/"
    path_postfix=".wav"
    table=pd.read_csv(tsv_file,sep='\t')
    table['wav_filesize'] = 0
    for i in range(table.path.size):
        table.path[i] = path_prefix + table.path[i] + path_postfix
        audio_len = convert_then_write_as_wav(table.path[i])
        table.wave_filesize = audio_len
    
    table.rename(columns={'path':'wav_filename','sentence':'transcript'}, inplace=True)
    table = table[['wav_filename','wav_filesize','transcript']]
    csv_file = tsv_file.split(".")[0] + ".csv"
    table.to_csv(csv_file, sep=",", encoding='utf-8')
    print('tsv_file ' + tsv_file + " is completed.")
print('all files reconfigured.')


#########################$$ISSUE NOW IS MP3 READING.....