import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

#from src.models.encoder.transformer_encoder import TransformerEncoder
from src.models.encoder.transformer_encoder_mytokenizer import TransformerEncoder
from src.models.encoder.acoustic_encoder import AcousticEncoder
from src.models.encoder.timing_encoder import TimingEncoder
from src.models.encoder.timing_encoder2 import TimingEncoder2
from src.models.vad.vad import VAD
from src.models.f0.f0 import F0
from src.models.vad_and_f0.vad_and_f0 import VAD_AND_F0

torch.autograd.set_detect_anomaly(True)


class FeatureExtractor(nn.Module):

    def __init__(self, config, device, is_use_silence=True, is_use_n_word=False):
        super().__init__()
        
        self.device = device
        self.config = config
        self.is_use_silence = is_use_silence
        self.is_use_n_word = is_use_n_word
        
        self.create_models(config)

    
    def create_models(self, config):
        
        ae = AcousticEncoder(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.acoustic_hidden_dim,
            self.config.model_params.acoustic_encoding_dim,
        )
        self.acoustic_encoder = ae
        te = TimingEncoder(
            self.config,
            self.device,
            self.config.model_params.timing_input_dim,
            self.config.model_params.timing_encoding_dim,
            is_use_silence=self.is_use_silence,
            is_use_n_word=self.is_use_n_word,
        )
        self.timing_encoder = te
        """
        te2 = TimingEncoder2(
            self.config,
            self.device,
            self.config.model_params.timing_input_dim,
            self.config.model_params.timing_encoding_dim,
            is_use_silence=self.is_use_silence,
            is_use_n_word=self.is_use_n_word,
        )
        self.timing_encoder2 = te2
        """
        
        # VAD
        """
        vad = VAD(self.device, self.config.model_params.input_dim, self.config.model_params.vad_hidden_dim)
        self.vad = vad
        self.vad.load_state_dict(torch.load(config.model_params.vad_model_path), strict=False)
        """
        # F0
        """
        f0 = F0(self.device, self.config.model_params.input_dim, self.config.model_params.vad_hidden_dim)
        self.f0 = f0
        self.f0.load_state_dict(torch.load(config.model_params.f0_model_path), strict=False)
        """
        # VAD_AND_F0
        """
        vad_and_f0 = VAD_AND_F0(self.device, self.config.model_params.input_dim, self.config.model_params.vad_hidden_dim)
        self.vad_and_f0 = vad_and_f0
        self.vad_and_f0.load_state_dict(torch.load(config.model_params.vad_and_f0_model_path), strict=False)
        """
        
        se = TransformerEncoder(
            self.config,
            self.device,
        )
        self.semantic_encoder = se

        
    def get_params(self):

        parameters = chain(
            self.acoustic_encoder.parameters(),
            self.timing_encoder.linear.parameters(),
            # self.timing_encoder2.linear.parameters(),
            self.semantic_encoder.parameters(),
        )
        return parameters

    
    def forward(self, specs, idxs, input_lengths, texts, indices, split):
        
        r_a = self.acoustic_encoder(specs, input_lengths)
        r_t = self.timing_encoder(specs, idxs, input_lengths, indices, split)
        # r_t, _ = self.timing_encoder2(specs, idxs, input_lengths, indices, split)
        # r_t, r_f0 = self.timing_encoder2(specs, idxs, input_lengths, indices, split)
        r_s = self.semantic_encoder(idxs, input_lengths)
        # r_vad = self.vad.get_features(specs, input_lengths)
        # r_f0 = self.f0.get_features(specs, input_lengths)
        # r_vad_and_f0 = self.vad_and_f0.get_features(specs, input_lengths)
        
        embs = torch.cat([r_s, r_a, r_t], dim=-1)
        # embs = torch.cat([r_s, r_a, r_t, r_f0], dim=-1)
        # embs = torch.cat([r_s, r_a, r_t, r_vad_and_f0], dim=-1)
     
        return embs
    
    
    def streaming_inference(self, feats, idxs, input_lengths, texts, indices, split, debug=False):
        
        r_a = self.acoustic_encoder(feats, input_lengths)
        # r_s = self.semantic_encoder(texts)
        r_s = self.semantic_encoder(idxs, input_lengths)  # from src.models.encoder.transformer_encoder_mytokenizer import TransformerEncoderの場合
        r_t = self.timing_encoder.streaming_inference(feats, idxs, input_lengths, indices, split, debug)
        if debug:
            r_t, silence, vad_preds = r_t
        
        embs = torch.cat([r_s, r_a, r_t], dim=-1)
        
        if debug:
            return embs, silence, vad_preds
        
        return embs
    
    
    def reset_state(self):
        self.acoustic_encoder.reset_state()
        self.timing_encoder.reset_state()
        # self.timing_encoder2.reset_state()
