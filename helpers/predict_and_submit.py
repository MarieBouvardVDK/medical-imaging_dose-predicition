import numpy as np
import torch
import time
import os
from tqdm import tqdm

def predict_and_submit_cuda(model, test_dataloader, submission=True, save_path=None):
    '''
    This function can be used to make predictions on the test set and create a submission folder for the competition. 
    Input: 
        - model: A trained model
        - test_dataloader: a dataloader containing the test data.
        - save_path: a path to a folder where the predictions will be saved.  
    '''
    print('Making predictions...')
    #moving to GPU if possible
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    #setting model to evaluation mode
    model.eval()
    
    with torch.no_grad():
      #iterating over dataloader
      for i, (image, name) in enumerate(tqdm(test_dataloader)):
          #moving image to cuda
          image = image.type(Tensor)
          start_predict = time.process_time()
          #making prediction
          predictions = model(image)
          #moving pred back to cpu
          predictions = predictions.cpu()
          #create submission file
          if submission == True: 
              save_name = save_path + name[0]
              np.save(save_name, predictions)
            
    print(f'Predictions made. Took {time.process_time() - start_predict}s.')


def predict_and_submit_mps(model, test_dataloader, submission=True, save_path = '/Users/mariebouvard/Desktop/medical-imaging/submission/'):
    '''
    This function can be used to make predictions on the test set and create a submission folder for the competition. 
    Input: 
        - model: A trained model
        - test_dataloader: a dataloader containing the test data.
        - save_path: a path to a folder where the predictions will be saved.  
    '''
    print('Making predictions...')
    #moving to GPU if possible
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    #setting model to evaluation mode
    model.eval()
    
    with torch.no_grad():
      #iterating over dataloader
      for i, (image, name) in enumerate(tqdm(test_dataloader)):
          #moving image to cuda
          image = image.to(device)
          start_predict = time.process_time()
          #making prediction
          predictions = model(image)
          #moving pred back to cpu
          predictions = predictions.cpu()
          #create submission file
          if submission == True: 
              save_name = save_path + name[0]
              np.save(save_name, predictions)
            
    print(f'Predictions made. Took {time.process_time() - start_predict}s.')