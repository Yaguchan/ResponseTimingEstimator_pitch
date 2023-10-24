import torch
import numpy as np
import os
from tqdm import tqdm


def train(model, optimizer, data_loader, device):
    
    model.train()
    
    correct_vad = 0.0
    ccounter = 0.0
    total_vad_loss = 0.0

    for batch in tqdm(data_loader):
        
        optimizer.zero_grad()
        outputs = model(batch, "train")
        loss = outputs["train_vad_loss"]
        loss.backward()
        optimizer.step()
        
        total_vad_loss += loss.detach().cpu().numpy()
        ccounter += 1
        correct_vad += outputs["train_vad_acc"]
            
    acc_vad = float(correct_vad) / ccounter
    loss_vad = float(total_vad_loss) / ccounter
    
    return loss_vad, acc_vad
 
   
def val(model, data_loader, deivce):
    model.eval()
    
    total = {"vad_loss":0, "vad_correct": 0}
    ccounter = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outputs = model(batch, "val")
            total["vad_loss"] += outputs["val_vad_loss"].detach().cpu().numpy()
            total["vad_correct"] += outputs["val_vad_acc"]
            ccounter += 1
            
        vad_acc = float(total["vad_correct"]) / ccounter
        vad_loss = float(total["vad_loss"]) / ccounter

    return vad_loss, vad_acc


def trainer(num_epochs, model, loader_dict, optimizer, device, outdir):

    best_val_vad_loss = 1000000000
    for epoch in range(num_epochs):
        print("Epoch:{}".format(epoch+1))
        train_vad_loss, train_vad_acc = train(model, optimizer, loader_dict['train'], device)
        val_vad_loss, val_vad_acc = val(model, loader_dict['val'], device)

        print('------------------------------------')
        print("Train VAD loss : {}".format(train_vad_loss))
        print("Train VAD      : {}".format(train_vad_acc))
        print('------------------------------------')
        print("Val VAD loss : {}".format(val_vad_loss))
        print("Val VAD      : {}".format(val_vad_acc))
        print('------------------------------------')
        if best_val_vad_loss > val_vad_loss:
            best_val_vad_loss = val_vad_loss
            torch.save(model.state_dict(), os.path.join(outdir, "best_val_loss_model.pth"))
        
            
def tester(model, loader_dict, device):
    test_vad_loss, test_vad_acc = val(model, loader_dict['test'], device)
    print('------------------------------------')
    print("Test VAD loss: {}".format(test_vad_loss))
    print("Test VAD     : {}".format(test_vad_acc))
    print('------------------------------------')