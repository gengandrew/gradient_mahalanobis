import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(128, 49),
            # nn.ReLU(inplace=True),
        )

        self.head = nn.Linear(49, 10)

        self.decoder = nn.Sequential(
            nn.Linear(49, 128),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),
            # nn.LogSoftmax(),
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        encoder_output = self.encoder(x)
        output = self.head(F.tanh(encoder_output))

        return output

    def encoder_forward(self, x):
        x = x.view(-1, 28*28)
        encoder_output = self.encoder(x)

        return encoder_output

    def decoder_forward(self, x):
        decoder_output = self.decoder(x)
        
        return decoder_output

    def autoencoder_forward(self, x):
        x = x.view(-1, 28*28)
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)

        return decoder_output