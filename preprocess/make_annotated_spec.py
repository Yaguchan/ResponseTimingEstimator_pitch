import os
import wave
import torch
import glob
import numpy as np
from tqdm import tqdm

import sflib.sound.sigproc.spec_image as spec_image
generator = spec_image.SpectrogramImageGenerator()


DATAROOT = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/ATR_Annotated/data_-500_2000'


def main():
    
    names_dir = os.path.join(DATAROOT, 'wav')
    names = os.listdir(names_dir)

    for i, name in enumerate(tqdm(names)):
        turn_dir = os.path.join(names_dir, name)
        turns = os.listdir(turn_dir)
        for j, turn in enumerate(turns):
            wav_path = os.path.join(turn_dir, turn)
            output_dir = turn_dir.replace('wav', 'spectrogram')
            os.makedirs(output_dir, exist_ok=True)
            spec_path =  os.path.join(output_dir, turn.replace('.wav', '_spectrogram.npy'))
            if os.path.exists(spec_path):
                continue
            wf = wave.open(wav_path)
            x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
            pad = np.zeros(int(16000*0.05), np.int16)
            x = np.concatenate([pad, x, pad])
            generator = spec_image.SpectrogramImageGenerator()
            spec = generator.input_wave(x)
            spec = np.vstack(spec)
            np.save(spec_path, spec)


if __name__ == '__main__':
    main()