import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from itertools import chain

torch.autograd.set_detect_anomaly(True)


class VAD(nn.Module):

    def __init__(self, device, input_dim, hidden_dim): #, silence_encoding_type="concat"):
        super().__init__()
        
        self.device = device
        
        self.c1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn3 = nn.BatchNorm2d(32)
        self.c4 = nn.Conv2d(32, 32, (7, 5), padding=(3, 2), stride=(3, 2))
        self.bn4 = nn.BatchNorm2d(32)
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
        
        self.reset_state()

    
    def forward(self, inputs, input_lengths):

        t = max(input_lengths)
        
        b, n, h, w = inputs.shape
        inputs = inputs.reshape(b*n, 1, h, w)
        inputs = self.bn1(F.relu(self.c1(inputs)))
        inputs = self.bn2(F.relu(self.c2(inputs)))
        inputs = self.bn3(F.relu(self.c3(inputs)))
        inputs = self.bn4(F.relu(self.c4(inputs))) 
        inputs = inputs.reshape(b, n, -1)
        
        inputs = rnn_utils.pack_padded_sequence(
            inputs, 
            input_lengths, 
            batch_first=True,
            enforce_sorted=False,
        )
        
        outputs, self.hidden_state = self.lstm(inputs, self.hidden_state)
        outputs, _ = rnn_utils.pad_packed_sequence(
            outputs, 
            batch_first=True,
            padding_value=0.,
            total_length=t,
        )
        
        logits = self.fc2(self.relu(self.fc1(outputs)))
        logits = logits.reshape(b, -1)
        
        return logits
    
    
    def get_features(self, inputs, input_lengths):
        
        with torch.no_grad():
        
            t = max(input_lengths)
            
            b, n, h, w = inputs.shape
            inputs = inputs.reshape(b*n, 1, h, w)
            inputs = self.bn1(F.relu(self.c1(inputs)))
            inputs = self.bn2(F.relu(self.c2(inputs)))
            inputs = self.bn3(F.relu(self.c3(inputs)))
            inputs = self.bn4(F.relu(self.c4(inputs))) 
            inputs = inputs.reshape(b, n, -1)
            
            inputs = rnn_utils.pack_padded_sequence(
                inputs, 
                input_lengths, 
                batch_first=True,
                enforce_sorted=False,
            )
            
            outputs, self.hidden_state = self.lstm(inputs, self.hidden_state)
            outputs, _ = rnn_utils.pad_packed_sequence(
                outputs, 
                batch_first=True,
                padding_value=0.,
                total_length=t,
            )
            
            feature_vad = self.fc1(outputs)
            feature_vad = feature_vad.reshape(b, t, -1)
        
        return feature_vad
    
    
    def reset_state(self):
        self.hidden_state = None

    
    def recog(self, inputs, input_lengths):
        outs = []
        with torch.no_grad():
            for i in range(len(input_lengths)):
                output = self.forward(inputs[i][:input_lengths[i]])
                outs.append(torch.sigmoid(output))            
        return outs


    def get_loss(self, probs, targets):
        return self.criterion(probs, targets.float())
    
    
    def get_acc(self, outputs, targets):
        pred = torch.round(torch.sigmoid(outputs))
        correct = (pred == targets).sum().float()
        acc = correct / targets.size(0)
        return acc.detach().cpu()
