import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain
from src.models.vad.vad import VAD

torch.autograd.set_detect_anomaly(True)


class VoiceActivityDetactor(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.create_models()

    
    def create_models(self):
        vad = VAD(
            self.device,
            self.config.model_params.input_dim,
            self.config.model_params.hidden_dim,
        )
        self.vad = vad

    
    def configure_optimizer_parameters(self):

        parameters = chain(
            self.vad.parameters(),
        )
        return parameters

    
    def forward(self, batch, split='train'):
        chs = batch[0]        
        vad_labels = batch[1].to(self.device)
        specs = batch[5].to(self.device)
        input_lengths = batch[6]
        batch_size = int(len(chs))
        self.vad.reset_state()
        
        vad_loss, vad_acc = 0, 0
        outputs_vad = self.vad(specs, input_lengths)
        for i in range(batch_size):
            output_vad = outputs_vad[i]
            vad_loss = vad_loss + self.vad.get_loss(output_vad[:input_lengths[i]], vad_labels[i][:input_lengths[i]])
            vad_acc = vad_acc + self.vad.get_acc(output_vad[:input_lengths[i]], vad_labels[i][:input_lengths[i]])
        vad_loss = vad_loss / float(batch_size)
        vad_acc = vad_acc / float(batch_size)

        outputs = {
            f'{split}_vad_loss': vad_loss,
            f'{split}_vad_acc': vad_acc,
        }

        return outputs