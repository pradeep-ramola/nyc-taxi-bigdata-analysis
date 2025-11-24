import torch
import numpy as np

def load_model(path):
    model = torch.load(path, map_location=torch.device("cpu"))
    model.eval()
    return model

def preprocess_input(data_dict):
    """
    Converts JSON input into tensor model expects
    Example fields: distance, duration, passenger_count, etc
    """
    values = np.array(list(data_dict.values()), dtype=np.float32)
    return torch.tensor(values).unsqueeze(0)
