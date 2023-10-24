import os
import pandas as pd

# python scripts/dataset/get_data_csv.py
PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/ATR_Annotated/data_-500_2000/csv'

def main(): 
    names = [name for name in os.listdir(os.path.join(PATH))]
    timing = [0 for i in range(11)]
    wav_num = []
    for name in names:
        df = pd.read_csv(os.path.join(PATH, name))
        wav_num.append(f'{len(df)}: {name}')
        for i, offsets in enumerate(df['offset']):
            # 発話タイミング分布
            if offsets < -250: timing[0] += 1
            elif -250 <= offsets < 0: timing[1] += 1
            elif 0 <= offsets < 250: timing[2] += 1
            elif 250 <= offsets < 500: timing[3] += 1
            elif 500 <= offsets < 750: timing[4] += 1
            elif 750 <= offsets < 1000: timing[5] += 1
            elif 1000 <= offsets < 1250: timing[6] += 1
            elif 1250 <= offsets < 1500: timing[7] += 1
            elif 1500 <= offsets < 1750: timing[8] += 1
            elif 1750 <= offsets < 2000: timing[9] += 1
            else: timing[10] += 1
            if 1750 <= offsets < 2000: print(name, i+1, offsets)
    print(timing)
    for s in sorted(wav_num):
        print(s)

if __name__ == '__main__':
    main()