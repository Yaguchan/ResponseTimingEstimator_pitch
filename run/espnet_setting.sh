# path
espnet_path=/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/save_github
cuda_root=/usr/local/cuda

# espnet install
git clone https://github.com/espnet/espnet
cd "${espnet_path}/espnet/tools"
ln -s /mnt/aoni04/higuchi/work/20200819/kaldi kaldi
. ./setup_cuda_env.sh /usr/local/cuda
./setup_anaconda.sh anaconda espnet 3.7
make

# run
cd "${espnet_path}/espnet/tools" 
bash -c ". activate_python.sh; . ./setup_cuda_env.sh ${cuda_root}; ./installers/install_warp-transducer.sh"
cd "${espnet_path}/espnet/egs2"
TEMPLATE/asr1/setup.sh atr/asr1
cd "${espnet_path}/espnet/egs2/atr/asr1"
cp -r /mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/atr6/asr1/data .
cp /mnt/aoni04/yaguchi/code/espnet/espnet2/asr/encoder/contextual_block_parallel_conformer_encoder.py ${espnet_path}/espnet/espnet2/asr/encoder/contextual_block_parallel_conformer_encoder.py
cp /mnt/aoni04/yaguchi/code/espnet/espnet2/asr/encoder/contextual_block_bifurcation_conformer_encoder.py ${espnet_path}/espnet/espnet2/asr/encoder/contextual_block_bifurcation_conformer_encoder.py
cp /mnt/aoni04/yaguchi/code/espnet/egs2/atr/asr1/asr.sh ${espnet_path}/espnet/egs2/atr/asr1/asr.sh
cp /mnt/aoni04/yaguchi/code/espnet/espnet2/tasks/asr.py ${espnet_path}/espnet/espnet2/tasks/asr.py
cp /mnt/aoni04/yaguchi/code/espnet/espnet/nets/pytorch_backend/transformer/attention.py ${espnet_path}/espnet/espnet/nets/pytorch_backend/transformer/attention.py
cp /mnt/aoni04/yaguchi/code/espnet/espnet2/bin/asr_transducer_inference.py ${espnet_path}/espnet/espnet2/bin/asr_transducer_inference.py 
cp /mnt/aoni04/yaguchi/code/espnet/espnet2/asr/espnet_model.py ${espnet_path}/espnet/espnet2/asr/espnet_model.py
cp /mnt/aoni04/yaguchi/code/espnet/espnet/nets/beam_search_transducer_online.py ${espnet_path}/espnet/espnet/nets/beam_search_transducer_online.py
cp /mnt/aoni04/yaguchi/code/espnet/espnet2/asr/transducer/utils.py ${espnet_path}/espnet/espnet2/asr/transducer/utils.py
mkdir myconf
mkdir myscripts
cp /mnt/aoni04/yaguchi/code/espnet/egs2/atr/asr1/myscripts/run_parallel_cbs_transducer.sh ${espnet_path}/espnet/egs2/atr/asr1/myscripts/run_parallel_cbs_transducer.sh
cp /mnt/aoni04/yaguchi/code/espnet/egs2/atr/asr1/myscripts/run_bifurcation_cbs_transducer.sh ${espnet_path}/espnet/egs2/atr/asr1/myscripts/run_bifurcation_cbs_transducer.sh
cp /mnt/aoni04/yaguchi/code/espnet/egs2/atr/asr1/myconf/train_asr_parallel_cbs_transducer.yaml ${espnet_path}/espnet/egs2/atr/asr1/myconf/train_asr_parallel_cbs_transducer.yaml
cp /mnt/aoni04/yaguchi/code/espnet/egs2/atr/asr1/myconf/train_asr_bifurcation_cbs_transducer.yaml ${espnet_path}/espnet/egs2/atr/asr1/myconf/train_asr_bifurcation_cbs_transducer.yaml
cp /mnt/aoni04/yaguchi/code/espnet/egs2/atr/asr1/myconf/decode_rnnt_conformer_streaming.yaml ${espnet_path}/espnet/egs2/atr/asr1/myconf/decode_rnnt_conformer_streaming.yaml
cp /mnt/aoni04/yaguchi/code/espnet/egs2/atr/asr1/myconf/train_lm.yaml ${espnet_path}/espnet/egs2/atr/asr1/myconf/train_lm.yaml
# ./myscripts/run_parallel_cbs_transducer.sh 