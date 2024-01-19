import os
import wave
import array
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


# python preprocess/make_noisewav.py
SNR = 20
DATAROOT = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/ATR_Annotated/data_-500_2000'
OUTDIR = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/ATR_Annotated/data_noise_-500_2000'
NOISEDIR = '/mnt/aoni04/kayanuma/data/JEIDA_NOISE'


# noise -> base_noise
N_TO_BASEN = {
    '02':'01',
    '03':'01',
    '05':'04',
    '07':'06',
    '09':'08',
    '11':'10',
    '13':'12',
    '14':'12',
    '16':'15',
    '18':'17',
    '20':'19',
    '22':'21',
    '24':'23',
    '26':'25',
    '28':'27',
    '30':'29',
    '32':'31',
    '34':'33',
    '36':'35',
    '38':'37',
    '40':'39',
    '42':'41',
    '43':'41',
    '45':'44',
    '47':'46'
}
NOISE_SPLIT = [['02', '03', '09', '11'], ['13', '14', '16', '18'], ['20', '22', '24'], ['26', '28', '30'], ['36', '38', '40'], ['42', '43', '47']]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def read_wave(filename):
    with wave.open(filename, 'r') as wave_file:
        params = wave_file.getparams()
        frames = wave_file.readframes(params.nframes)
        data = np.frombuffer(frames, dtype=np.int16)
    return data, params


def cal_amp(audio):
    amp = (np.frombuffer(audio, dtype="int16")).astype(np.float64)
    return amp


def cal_mean_amp(audio, start, end, rate=16000):
    audio2 = audio[start*(rate//1000):end*(rate//1000)]
    amp = (np.frombuffer(audio2, dtype="int16")).astype(np.float64)
    amp = np.sqrt(np.mean(np.square(amp), axis=-1))
    return amp


def cal_ratio(As, An, snr):
    An2 = As / (10 ** (snr/20))
    ratio = An2 / An
    return ratio


def main():
    seed_everything(1234)
    wav_dir = os.path.join(DATAROOT, 'wav')
    csv_dir = wav_dir.replace('wav', 'csv')
    csv2_dir = wav_dir.replace('wav', 'wav_start_end')
    output_dir = os.path.join(OUTDIR, f'SNR{SNR}', 'wav')
    names = os.listdir(wav_dir)
    random.shuffle(names)
    for i, name in enumerate(tqdm(names)):
        noise_set = i // 15
        noise_len = len(NOISE_SPLIT[noise_set])
        wav_name_dir = os.path.join(wav_dir, name)
        csv_path = os.path.join(csv_dir, name+'.csv')
        csv_df = pd.read_csv(csv_path)
        csv2_path = os.path.join(csv2_dir, name+'.csv')
        csv2_df = pd.read_csv(csv2_path)
        output_name_dir = os.path.join(output_dir, name)
        os.makedirs(output_name_dir, exist_ok=True)
        turns = os.listdir(wav_name_dir)
        for j, turn in enumerate(turns):
            # 
            wav_path = os.path.join(wav_name_dir, turn)
            noise_num = NOISE_SPLIT[noise_set][j % noise_len]
            noise_base_path = os.path.join(NOISEDIR, 'base_noise', f'{N_TO_BASEN[noise_num]}.wav')
            noise_path = os.path.join(NOISEDIR, 'noise_mono', f'{noise_num}.wav')
            output_path = os.path.join(output_name_dir, turn)
            # 
            clean_wav, clean_params = read_wave(wav_path)
            noise_base_wav, _ = read_wave(noise_path)
            noise_wav, _ = read_wave(noise_path)
            noise_wav = np.resize(noise_wav, clean_wav.shape)
            # 
            clean_amp = cal_amp(clean_wav)
            noise_amp = cal_amp(noise_wav)
            clean_mean_amp = cal_mean_amp(clean_wav, csv_df['start'].iloc[j-1]-csv2_df['wav_start'].iloc[j-1], csv_df['end'].iloc[j-1]-csv2_df['wav_start'].iloc[j-1])
            noise_mean_amp = cal_mean_amp(noise_base_wav, 0, len(noise_base_wav))
            # 
            ratio = cal_ratio(clean_mean_amp, noise_mean_amp, SNR)
            output = clean_amp + ratio * noise_amp
            output = np.clip(output, -32768, 32767)
            noisy_wave = wave.Wave_write(output_path)
            noisy_wave.setparams(clean_params)
            noisy_wave.writeframes(array.array('h', output.astype(np.int16)).tobytes())
            noisy_wave.close()
            

if __name__ == '__main__':
    main()