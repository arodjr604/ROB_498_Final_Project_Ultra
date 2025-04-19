import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, inputs):
        inputs = [lateral_conv(f) for f, lateral_conv in zip(inputs, self.lateral_convs)]
        for i in range(len(inputs) - 1, 0, -1):
            inputs[i - 1] += F.interpolate(inputs[i], size=inputs[i - 1].shape[2:], mode='nearest')
        outputs = [output_conv(f) for f, output_conv in zip(inputs, self.output_convs)]
        return outputs