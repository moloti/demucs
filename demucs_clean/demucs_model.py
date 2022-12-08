
import math

import torch as th
from torch import nn

import pytorch_lightning as pl
from .resample import downsample2, upsample2
from torch.nn import functional as F

class BLSTM(pl.LightningModule):
    def __init__(self, dim, layers, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        # Introduction of hidden state
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class Demucs(pl.LightningModule):
    @capture_init
    def __init__(
            self,
            sources,
            audio_channels,
            channels,
            depth,
            kernel_size,
            stride,
            growth,
            resample,
            glu,
            max_hidden,
            rescale,
            normalize,
            floor,
            sample_rate,
            lstm_layers,
            **kwargs):
        """
        Demucs model.

        Args:
            - sources (int): number of sources to separate.
            - audio_channels (int): stereo or mono.
            - channels (int): number of input hidden channels.
            - depth (int): number of layers.
            - kernel_size (int): kernel size.
            - stride (int): stride.
            - growth (float): number of channels is multiplied by this for every layer.
            - resample (int): amount of resampling to apply to the input/output. Can be one of 1, 2 or 4.
            - glu (bool): use GLU activation.
            - max_hidden(int): maximum number of channels. Can be useful to control the size/speed of the model.
            - rescale (float): controls custom weight initialization.
            - normalize (bool): normalize the input.
            - floor (float): stability flooring when normalizing.
            - sample_rate (float): sample_rate used for training the model.
            - lstm_layers (int): number of LSTM layers.
        """

        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.channels = channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.growth = growth
        self.resample = resample
        self.glu = glu
        self.max_hidden = max_hidden
        self.rescale = rescale
        self.normalize = normalize
        self.floor = floor
        self.sample_rate = sample_rate
        self.lstm_layers = lstm_layers

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Set activation function
        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1

        input_channels = audio_channels
        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(input_channels, channels, kernel_size, stride),
                nn.ReLU(),
                # add 1x1 convolution to each encoder layer and a convolution to each decoder layer.
                nn.Conv1d(channels, channels * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []


            decode += [
                nn.Conv1d(channels, ch_scale * channels, 1), activation,
                # No upsampling used
                nn.ConvTranspose1d(channels, output_channels, kernel_size, stride),
            ]

            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))

            output_channels = channels
            input_channels = channels
            channels = min(int(channels * growth), max_hidden)

        self.lstm = BLSTM(input_channels, layers=lstm_layers)
        if rescale:
            rescale_module(self, reference=rescale)



    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("demucs")
        parser.add_argument("--sources", type=int, default=2)
        parser.add_argument("--audio-channels", type=int, default=1)
        parser.add_argument("--channels", type=int, default=48)
        parser.add_argument("--depth", type=int, default=5)
        parser.add_argument("--kernel-size", type=int, default=8)
        parser.add_argument("--stride", type=int, default=4)
        parser.add_argument("--growth", type=float, default=2)
        parser.add_argument("--resample", type=int, default=4)
        parser.add_argument("--glu", type=bool, default=True)
        parser.add_argument("--max-hidden", type=int, default=10000)
        parser.add_argument("--rescale", type=float, default=0.1)
        parser.add_argument("--normalize", type=bool, default=True)
        parser.add_argument("--floor", type=float, default=1e-3)
        parser.add_argument("--sample-rate", type=float, default=8000)
        parser.add_argument("--lstm_layers", type=int, default=2)
        return parent_parser

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x
    
    def training_step(self, *args: Any, **kwargs: Any):
        return super().training_step(*args, **kwargs)

