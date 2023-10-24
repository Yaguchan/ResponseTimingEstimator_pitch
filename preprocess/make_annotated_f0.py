import os
import sys
import numpy as np
import soundfile as sf
from tqdm import tqdm
from world4py.np import apis

# DATA
DATAROOT = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/ATR_Annotated/data_-500_2000'
# make_annotated_spec.py image_shift
IMAGE_SHIFT=5


def getf0(filename):
    x, fs = sf.read(filename, dtype='float64')
    f0, time_axis = apis.dio(x, fs, frame_period=5.0)
    start_frame = 5
    f0 = f0[start_frame::int(2*IMAGE_SHIFT)]
    f0 = np.float32(f0)
    return f0


def main():
    
    names_dir = os.path.join(DATAROOT, 'wav')
    names = os.listdir(names_dir)
    outdir = DATAROOT.replace('wav', 'f0')
    os.makedirs(outdir, exist_ok=True)
    
    for name in tqdm(names):
        turn_dir = os.path.join(names_dir, name)
        turns = os.listdir(turn_dir)
        for turn in turns:
            wav_path = os.path.join(turn_dir, turn)
            output_dir = turn_dir.replace('wav', 'f0')
            os.makedirs(output_dir, exist_ok=True)
            f0_path = os.path.join(output_dir, turn.replace('.wav', '_f0.npy'))
            frames = sf.info(wav_path).frames
            if frames < int(0.1 * 16000):
                continue
            f0 = getf0(wav_path)
            np.save(f0_path, f0)


if __name__ == '__main__':
    main()