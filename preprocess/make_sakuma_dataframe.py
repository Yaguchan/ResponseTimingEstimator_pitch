import os
import numpy as np
import pandas as pd
from tqdm import tqdm

DATAROOT="/mnt/aoni04/jsakuma/data/ATR-Trek/dataframe"
DATAROOT2="/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/ATR_Annotated/make_annotated_data/dataframe"
OUT_DATAROOT="/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/test"

MIN=-500
MAX=2000

def get_csv(name):
    path = os.path.join(DATAROOT, name+'.csv')
    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    names1 = list(set([name.replace('.csv', '') for name in os.listdir(DATAROOT)]))
    names2 = list(set([name.replace('.csv', '') for name in os.listdir(DATAROOT2)]))
    file_names = [name for name in names1 if name in names2]
    num_data = 0
    print(len(file_names))
    
    for file_name in tqdm(file_names):
        df = get_csv(file_name)
        
        # 発話タイミングの範囲
        df = df[(df['offset']>MIN) & (df['offset']<MAX)]        
        # 現話者: User, 次話者: エージェントの場合
        df = df[(df['spk']==0) & (df['next_spk']==1)]
        
        # print(len(df))
        num_data += len(df)
        
        out_dir = os.path.join(OUT_DATAROOT, 'data_{}_{}/csv'.format(MIN, MAX))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(OUT_DATAROOT, 'data_{}_{}/csv/{}.csv'.format(MIN, MAX, file_name))
        df.to_csv(out_path)
    
    print(num_data)