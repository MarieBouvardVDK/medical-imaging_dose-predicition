import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import time

##################
# MODEL TRAINING #
##################

def train(model, train_loader, num_epoch, optimizer, criterion, lr_scheduler=None):
    """Train a generator on its own.

    Args:
        train_loader: (DataLoader) a DataLoader wrapping the training dataset
        num_epoch: (int) number of epochs performed during training
        lr: (float) learning rate of the discriminator and generator Adam optimizers
        beta1: (float) beta1 coefficient of the discriminator and generator Adam optimizers
        beta2: (float) beta1 coefficient of the discriminator and generator Adam optimizers

    Returns:
        generator: (nn.Module) the trained generator
    """

    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if cuda:
        model = model.cuda()
        criterion.cuda()

    # ----------
    #  Training
    # ----------

    #looping through epochs
    for epoch in range(num_epoch):
      running_loss = []
      print('Epoch {}/{}'.format(epoch + 1, num_epoch))

      #looping through train dataloader
      for i, (input, dose) in enumerate(tqdm(train_loader)):
  
        # Inputs = ct + possible masks + structural masks
        input = input.type(Tensor)
        real_dose = dose.type(Tensor)

        # Remove stored gradients
        optimizer.zero_grad()

        # Generate dose mask from imputs
        pred_dose = model(input)

        # Compute the corresponding loss
        loss = criterion(pred_dose, real_dose)
        running_loss.append(loss)

        # Compute the gradient and perform one optimization step
        loss.backward()
        optimizer.step()
      
      #if learning rate scheduler: take step
      if lr_scheduler != None:
        lr_scheduler.step(loss)
      #compute mean loss
      mean_loss = sum(running_loss)/len(running_loss)
      print(f'Loss: {mean_loss:.2f}') 
      print('=' * 80)

    return model
  
####################
# MODEL EVALUATION #
####################

def mean_absolute_error(image_true, image_generated):
    """Helper funtion to compute mean absolute error.
    Input:
        - image_true: (Tensor) true image
        - image_generated: (Tensor) generated image
    Output:
        - mae: (float) mean squared error
    """
    return torch.abs(image_true - image_generated).mean()
  
def evaluate(generator, train_loader, val_loader):
    """Function to evaluate model on train and validation set
    Input:
        - generator: trained model to evalaute
        
    Output:
      - df: dataframe with mean MAE values
      - history: dictionary for metrics
    """
    #initializing lists to store values
    train_mae, val_mae = [], []
    history = {
        'train_mae': [],
        'val_mae': []
        }
    
    #moving to GPU if possible
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    #evaluation
    with torch.no_grad():
        #looping through train dataloader
        for i, (input, dose) in enumerate(tqdm(train_loader)):

            # Inputs: ct + possible mask + structural mask
            input = input.type(Tensor)
            real_dose = dose.type(Tensor)
            pred_dose = generator(input)
            #computing MAE
            mae = mean_absolute_error(real_dose, pred_dose).item()
            #appending MAE to list
            train_mae.append(mae)
            history['train_mae'].append(mae)
            
        #looping through validation dataloader
        for i, (input, dose) in enumerate(tqdm(val_loader)):

            # Inputs: ct + possible mask + structural mask
            input = input.type(Tensor)
            real_dose = dose.type(Tensor)
            pred_dose = generator(input)
            #computing MAE
            mae = mean_absolute_error(real_dose, pred_dose).item()
            #appending MAE to list
            val_mae.append(mae)
            history['val_mae'].append(mae)
            
        #dict with mean MAE values
        dic = {"Training set": sum(train_mae)/len(train_mae), "Validation Set": sum(val_mae)/len(val_mae)}
        #dataframe for output
        df = pd.DataFrame(dic, index=['MAELoss'])
        
    return df, history
  
  def train_and_eval(model, train_loader, val_loader, num_epoch, optimizer, criterion, lr_scheduler=None):
    print('Starting training...')
    start_train = time.process_time()
    generator = train(model, train_loader, num_epoch, optimizer, criterion, lr_scheduler)
    print(f'Training done. Took {time.process_time() - start_train}s.')

    print('Starting evalution...')
    start_eval = time.process_time()
    df = evaluate(generator, train_loader, val_loader)
    print(f'Training done. Took {time.process_time() - start_eval}s.')
    
    return df
