import torch
import torchaudio
import codecs
import wave
import numpy as np
import pandas as pd
import os
import glob
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_text_file(name):
    text_user_path = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/ATR/ATR-Trek/Text_User/{}_L.xls'.format(name)
    text_agent_path = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/ATR/ATR-Trek/Text_Agent/{}.csv'.format(name)
    df_text_user = pd.read_excel(text_user_path)
    df_text_agent = pd.read_csv(text_agent_path)
    return df_text_user, df_text_agent


def save_turn_wav(wavepath, outpath, start, end, rate=16000, bits_per_sample=16):   
    wf = wave.open(wavepath, 'r')
    ch = wf.getnchannels()
    width = wf.getsampwidth()
    fr = wf.getframerate()
    fn = wf.getnframes()
    x = wf.readframes(wf.getnframes()) #frameの読み込み
    x = np.frombuffer(x, dtype= "int16") #numpy.arrayに変換
    turn = x[start*(rate//1000):end*(rate//1000)]
    wf.close()
    torchaudio.save(filepath=outpath, src=torch.tensor([turn]), sample_rate=rate)


TRAIN_SIZE=0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
SEED = 0


# PATH設定
# names(train.txt, val.txt, test.txt) 
NAMEROOT = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/ATR2022/asr/names'
# wav_mono
WAVROOT = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/ATR2022/wav_mono'
# epsnet/egs2/atr/asr1/data
OUTDIR = '/mnt/aoni04/yaguchi/code/espnet-share/egs2/atr/asr1/data'
# wavとtextの対応
user_list = sorted(glob.glob('/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/ATR/ATR-Trek/Text_User/*_L.xls'))
agent_list = sorted(glob.glob('/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/ATR/ATR-Trek/Text_Agent/*.csv'))


file_names1 = [file_path.split('/')[-1].replace('_L.xls', '') for file_path in user_list] 
file_names2 = [file_path.split('/')[-1].replace('.csv', '') for file_path in agent_list]
file_names = sorted(list(set(file_names1) & set(file_names2)))


file_names_train, file_names_val_test = train_test_split(file_names, test_size=TEST_SIZE+VAL_SIZE, random_state=SEED)
file_names_val, file_names_test = train_test_split(file_names_val_test, test_size=TEST_SIZE/(VAL_SIZE+TEST_SIZE), random_state=SEED)
file_names_train = sorted(file_names_train)
file_names_val = sorted(file_names_val)
file_names_test = sorted(file_names_test)
file_dict = {'train': file_names_train, 'valid': file_names_val, 'test': file_names_test}


for split in ['train', 'valid', 'test']:
    path_names = os.path.join(NAMEROOT, split+'.txt')
    for j in tqdm(range(len(file_dict[split]))): 
        with open(path_names, mode='a') as f:
            f.write(file_dict[split][j]+'\n')


if __name__ == "__main__":
    
    # espnet/egs2/atr/asr1/data
    os.makedirs(OUTDIR, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        
        os.makedirs(os.path.join(OUTDIR, split), exist_ok=True)
        
        text_list = []
        utt2spk_list = []
        scp_list = []
        uttrs = []
        
        path_text = os.path.join(OUTDIR, '{}/text'.format(split))
        path_wavscp = os.path.join(OUTDIR, '{}/wav.scp'.format(split))
        path_utt2spk = os.path.join(OUTDIR, '{}/utt2spk'.format(split))
        
        file_names_list = file_dict[split]
        for idx, file_name in enumerate(tqdm(file_names_list)):
            df_text_user, df_text_agent = get_text_file(file_name)
            
            # wav_mono
            wavpath = os.path.join(WAVROOT, '{}_user.wav'.format(file_name))
            
            cnt = 0
            for i in range(len(df_text_user)):
                
                """
                if df_text_user['start'].iloc[i] != df_text_user['start'].iloc[i]:
                    continue
                start=int(df_text_user['start'].iloc[i])
                end=int(df_text_user['end'].iloc[i])
                """
                if df_text_user['Unnamed: 0'].iloc[i] != df_text_user['Unnamed: 0'].iloc[i]:
                    continue
                start=int(df_text_user['Unnamed: 0'].iloc[i])
                end=int(df_text_user['Unnamed: 3'].iloc[i])
                
                if end-start<2000:
                    continue
                    
                cnt += 1
                
                # text = df_text_user['text'].iloc[i]
                text = df_text_user['Unnamed: 5'].iloc[i]
                text = text.replace('　', '').replace('、', '').replace('。', '')

                spk_id = 'U{:04}'.format(idx)
                uttr_id = spk_id+'_'+file_name.replace('-', '_')+'_{:03}'.format(cnt)
                name = spk_id+'_'+file_name.replace('-', '_')+'_{:03}.wav'.format(cnt)

                wav_out_dir = os.path.join(OUTDIR, 'wav', file_name)
                os.makedirs(wav_out_dir, exist_ok=True)

                wav_out_path = os.path.join(wav_out_dir, name)
                save_turn_wav(wavpath, wav_out_path, 0, end)
                
                text_list.append(uttr_id+' '+text+'\n')
                utt2spk_list.append(uttr_id+' '+spk_id+'\n')
                scp_list.append(uttr_id+' '+wav_out_path+'\n')
                uttrs.append(uttr_id)
        
        idxs = np.argsort(uttrs)
        
        for j in tqdm(idxs):                 
            with open(path_text, mode='a') as f:
                f.write(text_list[j])
            
            with open(path_wavscp, mode='a') as f:
                f.write(scp_list[j])

            with open(path_utt2spk, mode='a') as f:
                f.write(utt2spk_list[j])