import numpy as np
import torch

def make_predictions(model, test_dataloader):
    predictions = model(test_dataloader)
    return predictions
