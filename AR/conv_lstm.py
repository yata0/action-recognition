import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels = input_dim + hidden_dim,
                            out_channels = hidden_dim * 4,
                            kernel_size = kernel_size,
                            padding = self.padding,
                            bias = bias
                                )

    def forward(self, input, cell_state):
        c_h, c_c = cell_state
        
        combined = torch.cat([input, c_h], dim=1)

        out = self.conv(combined)

        out_i, out_f, out_o, out_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(out_i)
        f = torch.sigmoid(out_f)
        o = torch.sigmoid(out_o)
        g = torch.tanh(out_g)

        next_c = c_c * f + i * g
        next_h = torch.tanh(c_h) * o

        return next_h, next_c


    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.input_size[0], self.input_size[1])),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.input_size[0], self.input_size[1])))    

class ConvLSTM