import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import time

torch.set_default_dtype(torch.float32)

##################
# MODEL TRAINING #
##################

def train(model, train_loader, num_epoch, optimizer, criterion, lr_scheduler=None):
    """This model trains a given model.
    Input:
        - model: (nn.Module) model to be trained
        - train_loader: (DataLoader) a DataLoader wrapping the training dataset
        - num_epoch: (int) number of epochs performed during training
        - optimizer: optimizer to be used for training
        - criterion: loss to be used for training
    Output:
        - model: (nn.Module) the trained model
    """

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    #Tensor = torch.cuda.FloatTensor if mps else torch.FloatTensor
    
    #put model on GPU if possible
    if device == "mps":
        model = model.to(device)
        criterion.to(device)

    #looping through epochs
    for epoch in range(num_epoch):
        #initializing list to store loss
        running_loss = []
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))

        #looping through train dataloader
        for i, (input, real_dose) in enumerate(tqdm(train_loader)):
    
            # Inputs = ct + possible masks + structural masks
            input, real_dose = input.to(device), real_dose.type(torch.float32).to(device)

            #removing stored gradients
            optimizer.zero_grad()

            #generating dose mask from imputs
            pred_dose = model(input)

            #computing the corresponding loss
            loss = criterion(pred_dose, real_dose)
            #add loss to list
            running_loss.append(loss)

            #computing the gradient and performing one optimization step
            loss.backward()
            optimizer.step()

        #if learning rate scheduler: take step
        if lr_scheduler != None:
            lr_scheduler.step(loss)
        
        #computing average loss for epoch
        mean_loss = sum(running_loss)/len(running_loss)
        #print metric
        print(f'Loss: {mean_loss:.2f}') 
        print('-' * 50)

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
 

def evaluate_generator(generator, train_loader, val_loader):
    '''
    Function to evaluate model on train and validation set
    Input:
        - generator: trained model to evaluate
        
    Output:
      - df: dataframe with mean MAE values
    '''
    
    #initializing lists to store values
    train_mae, val_mae = [], []
    history = {
        'all_train_mae': [],
        'mean_train_mae': [],
        'all_val_mae': [],
        'mean_val_mae': []
        }
    
    #moving to GPU if possible
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    #Tensor = torch.cuda.FloatTensor if mps else torch.FloatTensor
    
    #evaluation
    with torch.no_grad():
        #looping through train dataloader
        print('Evaluation on Training Set')
        for i, (input, real_dose) in enumerate(tqdm(train_loader)):

            # Inputs: ct + possible mask + structural mask
            input, real_dose = input.to(device), real_dose.type(torch.float32).to(device)
            pred_dose = generator(input)
            #computing MAE
            mae = mean_absolute_error(real_dose, pred_dose).item()
            #appending MAE to list
            train_mae.append(mae)
            history['all_train_mae'].append(mae)

        mean_train = sum(train_mae)/len(train_mae)
        history['mean_train_mae'].append(mean_train)
        print(f'Mean Absolute Error on Training set is {mean_train:.2f}')

        print('-' * 50)

        #looping through validation dataloader
        print('Evaluation on Validation Set')
        for i, (input, real_dose) in enumerate(tqdm(val_loader)):

            # Inputs: ct + possible mask + structural mask
            input, real_dose = input.to(device), real_dose.type(torch.float32).to(device)
            pred_dose = generator(input)
            #computing MAE
            mae = mean_absolute_error(real_dose, pred_dose).item()
            #appending MAE to list
            val_mae.append(mae)
            history['all_val_mae'].append(mae)

        mean_val = sum(val_mae)/len(val_mae)
        history['mean_val_mae'].append(mean_val)
        print(f'Mean Absolute Error on Validation set is {mean_val:.2f}')
            
        #dict with mean MAE values
        dic = {"Training set": sum(train_mae)/len(train_mae), "Validation Set": sum(val_mae)/len(val_mae)}
        #dataframe for output
        df = pd.DataFrame(dic, index=['MAELoss'])
        
    return df, history

def train_and_eval(model, train_loader, val_loader, num_epoch, optimizer, criterion, lr_scheduler = None):
    '''
    Function to launch training and evaluation of model
    Input:
        - model:(nn.Module) model to evaluate
        - train_loader:(Dataloader) training dataloader
        - val_loader: (Dataloader) validation dataloader
        - num_epoch: (int) number of epochs for training
        - optimizer: (torch.optim) optimizer for training
        - criterion: loss for training
        - lr_scheduler: learning rate scheduler for training
     Output:
        - df: dataframe with performance results
        - model: (nn.Module) trained model
    '''

    print('Starting training...')
    print('-' * 50)
    start_train = time.process_time()
    generator = train(model, train_loader, num_epoch, optimizer, criterion, lr_scheduler)
    print(f'Training done. Took {time.process_time() - start_train:.2f}s, {(time.process_time() - start_train)/num_epoch:.2f}s per epoch.\n')

    print('=' * 50)

    print('Starting evalution...')
    print('-' * 50)
    start_eval = time.process_time()
    df, history = evaluate_generator(generator, train_loader, val_loader)
    print(f'Evaluation done. Took {time.process_time() - start_eval:.2f}s.')

    return generator, df, history




