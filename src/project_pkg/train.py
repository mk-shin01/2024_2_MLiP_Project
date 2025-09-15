"""Training loop and related routines extracted from the original notebook."""

from . import utils
from . import models, data

# ---- cell 1 ----
import time
import copy
import torch
import numpy as np

def train_model(model, criterion, optimizer, scheduler, num_epochs=40):
    """
    Support function for model training.
    Args:
    model: Model to be trained
    criterion: Optimization criterion (loss)
    optimizer: Optimizer to use for training
    scheduler: Instance of ``torch.optim.lr_scheduler``
    num_epochs: Number of epochs
    Return:
    model: most accurate model at accuracy measure
    <list>: epoch number iterations
    <<list>, <list>>: training loss and accuracy iterations
    <<list>, <list>>: val loss and accuracy iterations
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_lst = []
    trn_loss_lst = []
    trn_acc_lst = []
    val_loss_lst = []
    val_acc_lst = []

    for epoch in range(num_epochs):
        print('-' * 50)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data batches
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
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
                scheduler.step()

            epoch_acc = running_corrects.double() / len(data_loader.dataset)
            epoch_loss = running_loss / len(data_loader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # save training val metadata metrics
            if phase == 'train':
                epoch_lst.append(epoch)
                trn_loss_lst.append(np.round(epoch_loss, 4))
                trn_acc_lst.append(np.round(epoch_acc.cpu().item(), 4))

            elif phase == 'val':
                val_loss_lst.append(np.round(epoch_loss, 4))
                val_acc_lst.append(np.round(epoch_acc.cpu().item(), 4))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'model_best.pt')
                best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), 'model_latest.pt')

    trn_metadata = [trn_loss_lst, trn_acc_lst]
    val_metadata = [val_loss_lst, val_acc_lst]

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_lst, trn_metadata, val_metadata

# ---- cell 2 ----
model_ft, epc, trn, val = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=training_epochs)

