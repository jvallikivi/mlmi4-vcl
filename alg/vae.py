from typing import Union

import torch
from torch import nn
from torch.distributions import kl_divergence, Normal
import numpy as np
from tqdm.auto import tqdm


class Encoder(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, num_hidden=1, emb_dim=8):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(num_hidden):
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), nn.ReLU()))

        self.head_loc = nn.Linear(hidden_dim, emb_dim)
        self.head_scale = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x, num_samples=1):
        for layer in self.layers:
            x = layer(x)

        loc = self.head_loc(x)
        scale = torch.exp(self.head_scale(x))
        out = []
        for _ in range(num_samples):
            out.append(loc + torch.randn_like(scale) * scale)
        return out, kl_divergence(Normal(loc, scale), Normal(0, 1)).sum()

    def forward_MLE(self, x):
        for layer in self.layers:
            x = layer(x)

        out = self.head_loc(x)
        return out


class ConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels=3, image_width=32, image_height=32, emb_dim=8, conv_output_c=[32]):
        super(ConvolutionalEncoder, self).__init__()
        self.layers = nn.ModuleList()
        widths, heights = [image_width], [image_height]
        for i, output_channels in enumerate(conv_output_c):
            input_channels = in_channels if i == 0 else conv_output_c[i-1]
            kernel_size, stride = 4, 2
            self.layers.append(nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()))
            widths.append(
                int(np.floor(float(widths[-1] - kernel_size)/stride)) + 1)
            heights.append(
                int(np.floor(float(heights[-1] - kernel_size)/stride)) + 1)

        self.conv_output_c = conv_output_c
        self.widths = widths
        self.heights = heights

        self.flattened_size = int(conv_output_c[-1] * widths[-1] * widths[-1])
        self.head_loc = nn.Linear(self.flattened_size, emb_dim)
        self.head_scale = nn.Linear(self.flattened_size, emb_dim)

    def forward(self, x, num_samples=1):
        for layer in self.layers:
            x = layer(x)

        x = nn.Flatten(start_dim=0 if x.dim() == 3 else 1)(x)
        loc = self.head_loc(x)
        scale = torch.exp(self.head_scale(x))
        out = []
        for _ in range(num_samples):
            out.append(loc + torch.randn_like(scale) * scale)
        return out, kl_divergence(Normal(loc, scale), Normal(0, 1)).sum()

    def forward_MLE(self, x):
        for layer in self.layers:
            x = layer(x)

        out = self.head_loc(x)
        return out


class Decoder(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=256, num_hidden=1, out_dim=784):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(num_hidden):
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), nn.ReLU()))

        self.layers.append(
            nn.Linear(hidden_dim if num_hidden > 0 else in_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)


class ConvolutionalDecoder(nn.Module):
    def __init__(self, emb_dim=8, hidden_dim=512, widths=[32, 32], heights=[32, 32], transposed_conv_input_c=[32]):
        super(ConvolutionalDecoder, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Unflatten(
                1, (transposed_conv_input_c[0], heights[0], widths[0]))
        )

        self.deconv = nn.ModuleList()
        for i, input_channels in enumerate(transposed_conv_input_c[:-1]):
            output_channels = transposed_conv_input_c[i+1]
            stride = 2
            kernel_size_h = heights[i+1] - (heights[i] - 1) * stride
            kernel_size_w = widths[i+1] - (widths[i] - 1) * stride

            conv_t = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=(
                kernel_size_h, kernel_size_w), stride=stride)
            if i == len(transposed_conv_input_c) - 2:
                self.deconv.append(nn.Sequential(
                    conv_t,
                    nn.BatchNorm2d(output_channels),
                ))
            else:
                self.deconv.append(nn.Sequential(
                    conv_t,
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                ))

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.deconv:
            x = layer(x)

        return torch.sigmoid(x)


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_dim=784, hidden_dim_enc=256, num_hidden_enc=1, emb_dim=8, hidden_dim_dec=256, num_hidden_dec=1):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = Encoder(in_dim=in_dim, hidden_dim=hidden_dim_enc,
                               num_hidden=num_hidden_enc, emb_dim=emb_dim)
        self.decoder = Decoder(in_dim=emb_dim, hidden_dim=hidden_dim_dec,
                               num_hidden=num_hidden_dec, out_dim=in_dim)

    def forward(self, x):
        return self.encoder(x, 1)[0][0]

    def calculate_loss(self, x, num_samples=1):
        embeddings, KL = self.encoder(x, num_samples)
        loss = 0
        for embedding in embeddings:
            loss += torch.sum(torch.pow(x - self.decoder(embedding), 2.0))
        return loss / x.shape[0] / num_samples + KL / x.shape[0]

    def autoencode(self, x):
        return self.decoder(self(x))


class ConvolutionalVariationalAutoencoder(nn.Module):
    def __init__(self, in_channels=3, image_width=32, image_height=32, emb_dim=16):
        super(ConvolutionalVariationalAutoencoder, self).__init__()

        encoder_conv_layer_output_channels = [32, 64, 128]
        self.encoder = ConvolutionalEncoder(in_channels=in_channels, image_width=image_width,
                                            image_height=image_height, emb_dim=emb_dim, conv_output_c=encoder_conv_layer_output_channels)

        self.decoder = ConvolutionalDecoder(emb_dim=emb_dim,
                                            hidden_dim=self.encoder.flattened_size,
                                            transposed_conv_input_c=list(
                                                reversed(encoder_conv_layer_output_channels)) + [in_channels],
                                            widths=list(
                                                reversed(self.encoder.widths)),
                                            heights=list(
                                                reversed(self.encoder.heights)),
                                            )

    def forward(self, x):
        return self.encoder(x, 1)[0][0]

    def calculate_loss(self, x, num_samples=1):
        embeddings, KL = self.encoder(x, num_samples)
        loss = 0
        for embedding in embeddings:
            loss += torch.sum(torch.pow(x - self.decoder(embedding), 2.0))
        return loss / x.shape[0] / num_samples + KL / x.shape[0]

    def autoencode(self, x):
        return self.decoder(self(x[None, :, :, :]))


def train_variational_autoencoder(model: Union[VariationalAutoencoder, ConvolutionalVariationalAutoencoder], data, n_epochs=100, batch_size=256, verbose=False):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in ((lambda x: tqdm(x, desc='Training VAE', position=2, leave=False)) if verbose else iter)(range(n_epochs)):
        for batch in range(int(np.ceil(data.shape[0] / batch_size))):
            batch_idx0 = batch * batch_size
            batch_idx1 = batch * batch_size + batch_size
            opt.zero_grad()
            loss = model.calculate_loss(
                x=data[batch_idx0: batch_idx1], num_samples=1)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 5)
            opt.step()
