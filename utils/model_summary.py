from torchinfo import summary
import torch
def model_summary(model, input_size, device):
    summary(model, input_size=input_size, device=device)
    