import torch
import torch.nn as nn

class Conv1DAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for TESS Light Curve Anomaly Detection.
    Input shape:  (batch_size, 1, 2048)
    Bottleneck:  (batch_size, 32)
    Output shape: (batch_size, 1, 2048)
    """
    def __init__(self, bottleneck_dim=32):
        super(Conv1DAutoencoder, self).__init__()
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),   # -> (16, 1024)
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),  # -> (32, 512)
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),  # -> (64, 256)
            nn.LeakyReLU(0.2)
        )
        
        self.flatten_dim = 64 * 256
        self.fc_encoder = nn.Linear(self.flatten_dim, bottleneck_dim)
        self.fc_decoder = nn.Linear(bottleneck_dim, self.flatten_dim)
        
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1), # -> (32, 512)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1), # -> (16, 1024)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),  # -> (1, 2048)
            nn.Identity()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode
        x_enc = self.encoder_conv(x)
        x_flat = x_enc.view(batch_size, -1)
        bottleneck = self.fc_encoder(x_flat)
        
        # Decode
        x_dec = self.fc_decoder(bottleneck)
        x_reconstructed_input = x_dec.view(batch_size, 64, 256)
        reconstruction = self.decoder_conv(x_reconstructed_input)
        
        return reconstruction
