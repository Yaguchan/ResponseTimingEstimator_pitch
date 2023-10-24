import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# python scripts/test/timing.py
# DATAROOT = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/exp/annotated/data_-500_2000/timing/asj2023'
# DATAROOT = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/exp/annotated/data_-500_2000/timing'
DATAROOT = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/exp/annotated/data_-500_2000/timing/revised_loss'


# 
# EXP_NAME = ['baseline' ]
# EXP_NAME = ['model_proposed_mla_s1234', 'model_baseline_mla_s1234',]
EXP_NAME = ['model_baseline_mla_s1234_0', 'model_baseline_mla_s1234_1', 'model_baseline_mla_s1234_2']
# EXP_NAME = ['model_baseline_mla_s1234_loss0', 'model_baseline_mla_s1234_loss150_2', 'model_baseline_mla_s1234_loss250_2']
# EXP_NAME = ['model_baseline_mla_s1234_loss0', 'model_baseline_mla_s1234_loss1', 'model_baseline_mla_s1234_loss50', 'model_baseline_mla_s1234_loss100', 'model_baseline_mla_s1234_loss150', 'model_baseline_mla_s1234_loss200', 'model_baseline_mla_s1234_loss250', 'model_baseline_mla_s1234_loss500', 'model_baseline_mla_s1234_loss1000']
# EXP_NAME = ['model_baseline_mla_s1234_udas', 'model_baseline_mla_s1234_sda', 'model_baseline_mla_s1234_sdas']
# EXP_NAME = ['model_baseline_mla_s1234_linear_udas', 'model_baseline_mla_s1234_linear_sdas']#, 'model_baseline_mla_s1234_linear_uda', 'model_baseline_mla_s1234_linear_sda']
# EXP_NAME = ['model_baseline_mla_s1234_linguistic_transformer_6', 'model_baseline_mla_s1234_linguistic_transformer_12']
# EXP_NAME = ['model_baseline_mla_s1234_tg_1lstm_max3words'] #, 'model_baseline_mla_s1234_tg_2lstm_max3words', 'model_baseline_mla_s1234_tg_3lstm_max3words']
# EXP_NAME = ['model_baseline_mla_s1234_tg_plus_1lstm_max3words_avevalue', 'model_baseline_mla_s1234_tg_plus_2lstm_max3words_avevalue', 'model_baseline_mla_s1234_tg_plus_3lstm_max3words_avevalue']
# EXP_NAME = ['model_baseline_mla_s1234_tg_1lstm_max3words_softmax_ave'] #, 'model_baseline_mla_s1234_tg_2lstm_max3words_softmax_avevalue', 'model_baseline_mla_s1234_tg_3lstm_max3words_softmax_avevalue']
# EXP_NAME = ['model_baseline_mla_s1234_tg_1lstm_max3words_softmax_08', 'model_baseline_mla_s1234_tg_2lstm_max3words_softmax_08', 'model_baseline_mla_s1234_tg_3lstm_max3words_softmax_08']
# EXP_NAME = ['model_baseline_mla_s1234_tg_1lstm_max3words_softmax_no'] #, 'model_baseline_mla_s1234_tg_1lstm_max3words_softmax_02', 'model_baseline_mla_s1234_tg_1lstm_max3words_softmax_03', 'model_baseline_mla_s1234_tg_1lstm_max3words_softmax_04', 'model_baseline_mla_s1234_tg_1lstm_max3words_softmax_05', 'model_baseline_mla_s1234_tg_1lstm_max3words_softmax_06', 'model_baseline_mla_s1234_tg_1lstm_max3words_softmax_07', 'model_baseline_mla_s1234_tg_1lstm_max3words_softmax_08', 'model_baseline_mla_s1234_tg_1lstm_max3words_softmax_09', 'model_baseline_mla_s1234_tg_1lstm_max3words_softmax_10']
# EXP_NAME = ['model_baseline_mla_s1234_tg_1lstm_max1words', 'model_baseline_mla_s1234_tg_1lstm_max2words', 'model_baseline_mla_s1234_tg_1lstm_max3words', 'model_baseline_mla_s1234_tg_1lstm_max4words', 'model_baseline_mla_s1234_tg_1lstm_max5words'] 
# EXP_NAME = ['model_baseline_mla_s1234_tg_plus_1lstm_max1words' , 'model_baseline_mla_s1234_tg_plus_1lstm_max2words', 'model_baseline_mla_s1234_tg_plus_1lstm_max3words', 'model_baseline_mla_s1234_tg_plus_1lstm_max4words', 'model_baseline_mla_s1234_tg_plus_1lstm_max5words']
SEED_LIST = ["cv1", "cv2", "cv3", "cv4", "cv5"]


def main():
    
    recall, precision, f1, csv, csv2 = {}, {}, {}, {}, {}
    
    for name in EXP_NAME:
        recall[name] = np.zeros(201)
        precision[name] = np.zeros(201)
        f1[name] = np.zeros(201)
        csv[name] = []    
    
    for seed in SEED_LIST:
        for name in EXP_NAME:
            
            recall[name] += np.load("{}/{}/npy/recall_{}.npy".format(DATAROOT, name, seed))
            precision[name] += np.load("{}/{}/npy/precision_{}.npy".format(DATAROOT, name, seed))
            f1[name] += np.load("{}/{}/npy/f1_{}.npy".format(DATAROOT, name, seed))
            
            if len(csv[name])>0:
                csv_tmp = pd.read_csv("{}/{}/csv/df_{}.csv".format(DATAROOT, name, seed))
                csv_tmp['exp'] = [seed]*len(csv_tmp)
                csv[name] = pd.concat([csv[name], csv_tmp])
            else:
                csv[name] = pd.read_csv("{}/{}/csv/df_{}.csv".format(DATAROOT, name, seed))        
                csv[name]['exp'] = [seed]*len(csv[name])     

    for name in EXP_NAME:
        recall[name] /= len(SEED_LIST)
        precision[name] /= len(SEED_LIST)
        f1[name] /= len(SEED_LIST)
        
    metrics_list = [precision, recall, f1]
    err_list = np.arange(0, 3050, 50)

    print('{}: {:d}ms, {:d}ms, {:d}ms'.format('metrics'.ljust(9), 50*5, 50*10, 50*20))
    for name in EXP_NAME:
        print(name)
        for i, metrics in enumerate(['precision', 'recall', 'f1']):
            print('{}: {:.3f}, {:.3f}, {:.3f}'.format(metrics.ljust(9), metrics_list[i][name][5], metrics_list[i][name][10], metrics_list[i][name][20]))   
        print()

if __name__ == '__main__':
    main()