# ResponseTimingEstimator_pitch

## モデルの学習
### VAD
```
python scripts/vad/run_annotated_vad.py configs/vad/annotated_vad.json --gpuid 0
```

### F0推定器
```
python scripts/f0/run_annotated_f0.py configs/f0/annotated_f0.json --gpuid 0
```

### 発話タイミング推定器
```
python scripts/timing/run_annotated_timing.py configs/timing/annotated_timing_baseline_mla_s1234.json --model baseline --gpuid 0 --cv_id 1
```