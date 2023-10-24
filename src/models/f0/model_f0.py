import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

from src.models.f0.f0 import F0

torch.autograd.set_detect_anomaly(True)


class F0_Predictor(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.R = 60
        
        self.create_models()

    
    def create_models(self):
        f0 = F0(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.vad_hidden_dim,
        )
        self.f0 = f0

    
    def configure_optimizer_parameters(self):

        parameters = chain(
            self.f0.parameters(),
        )
        return parameters

    
    def forward(self, batch, split='train'):
        chs = batch[0]
        specs = batch[5].to(self.device)
        input_lengths = batch[6]
        f0_labels = batch[9].to(self.device)
        batch_size = int(len(chs))

        self.f0.reset_state()
        
        f0_loss = 0
        outputs_f0 = self.f0(specs, input_lengths)
        for i in range(batch_size):
            output_f0 = outputs_f0[i]
            loss_mask = (f0_labels[i] > 0).int()
            f0_loss = f0_loss + self.f0.get_loss(output_f0[:input_lengths[i]]*loss_mask[:input_lengths[i]], f0_labels[i][:input_lengths[i]]) / torch.sum(loss_mask[:input_lengths[i]])
        f0_loss = f0_loss / float(batch_size)

        outputs = {
            f'{split}_f0_loss': f0_loss,
        }

        return outputs