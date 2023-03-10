import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pandas as pd

##############
#MODEL TRAINING
###############

def train(model, train_loader, num_epoch, optimizer, criterion):
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

    cuda = True if torch.cuda.is_available() else False
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    #put model on GPU if possible
    if cuda:
        model = model.cuda()
        criterion.cuda()

    #looping through epochs
    for epoch in range(num_epoch):
      #initializing list to store loss
      running_loss = []
      print('Epoch {}/{}'.format(epoch + 1, num_epoch))

      #looping through train dataloader
      for i, (input, dose) in enumerate(tqdm(train_loader)):
  
        # Inputs = ct + possible masks + structural masks
        input = input.type(Tensor)
        real_dose = dose.type(Tensor)

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
      
      #computing average loss for epoch
      mean_loss = sum(running_loss)/len(running_loss)
      #print metric
      print(f'Loss: {mean_loss:.2f}') 
      print('=' * 80)

    return model
 
##############
#MODEL EVALUATION
###############
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
    """Function to evaluate model on train and validation set

    Input:
        - generator: trained model to evalaute
        
    Output:
      - df: dataframe with mean MAE values
    """
    #initializing lists to store values
    train_mae, val_mae = [], []
    
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
            
        #dict with mean MAE values
        dic = {"Training set": sum(train_mae)/len(train_mae), "Validation Set": sum(val_mae)/len(val_mae)}
        #dataframe for output
        df = pd.DataFrame(dic, index=['MAELoss'])
        
    return df
