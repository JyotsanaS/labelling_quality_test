import os
from torch.utils.data import Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
import wandb
import torch
import os
from transformers.models.dinov2 import Dinov2Config, Dinov2Model
from torch.optim import lr_scheduler
import torch.nn as nn
from adamwr.adamw import AdamW
from dataset import CustomDataset
from model import DinoV2_6channels

def train_model(model, criterion, optimizer, scheduler, num_epochs=200):
    model = model.to(device)
    since = time.time()
    # Add three dense layers for classification into 10 classes
    
    for epoch in range(100, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.float().to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step(running_loss / dataset_sizes[phase])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}, {phase} Accuracy: {epoch_acc:.4f}')
            
            # Log the loss to Weights and Biases after every epoch
            wandb.log({"Epoch": epoch,
                        f"{phase} Loss": epoch_loss,
                        f"{phase} Accuracy": epoch_acc})

            if phase == 'val':
                checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': scheduler.state_dict()}

                epoch_model_path = f'/home/ubuntu/home/ubuntu/6channel_training/dinov2_checkpoints/checkpoint_epoch_{epoch}_loss_{epoch_loss}.pth'
                torch.save(checkpoint, epoch_model_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model

if __name__ == "__main__":
    data_dir = "/home/ubuntu/home/ubuntu/6channel_training/Euro6datasetabs"
    image_datasets = {x: CustomDataset(os.path.join(data_dir, x))
                    for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dinov2 = DinoV2_6channels()
    dinov2 = torch.load('./dinov2_models_train1/model_epoch_99_loss_0.143072206396956.pt')
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = AdamW(dinov2.parameters(), lr=1e-3, weight_decay=1e-5)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min')
    
    # Initialize Weights and Biases
    wandb.login()
    wandb.init(name='dinov2_training_6channels_100_200', 
            project='Eurosat_6channel_training',
            notes='Eurosat data', 
            tags=['satsure', 'training','Eurosat'])

    train_model(dinov2, criterion, optimizer_ft, exp_lr_scheduler)
