import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, hidden_size*4)
    
    def forward(self, x, current_state):
        h_current, c_current = current_state
        combined = torch.cat([x, h_current], dim=1)
        out = self.linear(combined)
        out_i, out_f, out_o, out_g = torch.split(combined)
        i = torch.sigmoid(out_i)
        f = torch.sigmoid(out_f)
        o = torch.sigmoid(out_o)
        g = torch.tanh(out_g)

        new_cell_state = i * g + c_current * f
        new_hidden_state = o * torch.tanh(new_cell_state)
        return new_hidden_state, new_cell_state   