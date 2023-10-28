import torch
import torch.nn as torch
from typing import List
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], activations: List[str]):

        assert len(hidden_sizes) == len(activations)
        super(LSTM, self).__init__()
       # self.LSTM = nn.LSTM()

       
