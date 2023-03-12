import numpy as np
import torch
import time
import os
import tqdm

def predict_and_submit(model, test_dataloader, submission=True, save_path = '/Users/mariebouvard/Desktop/medical-imaging/submissions'):
    print('Making predictions...')
    with torch.no_grad():
        for i, (image, name) in enumerate(tqdm(test_dataloader)):
            start_predict = time.process_time()
            predictions = model(image)
            if submission == True: 
                save_name = save_path + name[0]
                np.save(save_name, predictions)
    print(f'Predictions made. Took {time.process_time() - start_predict}s.')

    return predictions
