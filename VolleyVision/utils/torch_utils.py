import torch

def select_device(device=''):
    """Seleciona o dispositivo (CPU/GPU)"""
    if device.lower() == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')