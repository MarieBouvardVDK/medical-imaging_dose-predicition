import numpy as np
import torch
import time
import os
import tqdm

def predict_and_submit(model, test_dataloader, submission=True, save_path = path_for_pred):
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

    return predictions
