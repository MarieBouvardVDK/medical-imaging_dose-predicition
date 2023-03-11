import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

####################
# Dose image visualization
####################
def visualize_dose(model, dataloader):
  """
  This function allows to visualize Real VS Predicted Dose
  Input:
    - model: (nn.Module) trained model to use to make prediction
    - dataloader: (Dataloader) dataloader with data to use to make prediction
  """
  #moving to GPU if possible
  cuda = True if torch.cuda.is_available() else False
  Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

  #setting model to evaluation mode
  model.eval()

  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

  with torch.no_grad():
    #collecting data sample
    input, real_dose = next(iter(dataloader))
    #moving to gpu
    input = input.type(Tensor)
    real_dose = real_dose.type(Tensor)
    #making prediction
    pred_dose = model(input)

    #adapting dose data
    real_dose = real_dose.cpu().numpy()[0,0]#.transpose((0, 2, 3, 1))
    pred_dose = pred_dose.cpu().numpy()[0,0]#.transpose((0, 2, 3, 1))
    # print(real_dose[0][0].shape)
    # print(pred_dose[0][0].shape)

    axes[0].imshow(real_dose, cmap='gray')
    axes[0].set_title('Real Dose')
    axes[0].axis('off')
    
    axes[1].imshow(pred_dose, cmap='gray')
    axes[1].set_title('Predicted Dose')
    axes[1].axis('off')

  plt.show()
