
"""
FCN
This model is for MRI segmentation, Binary segmentation (tumour vs non-tumour), 4 MRI channels
"""
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        # ----------------------------------------------
        # Encoder 

        ## First convolution layer, Input: 4 MRI channels, Output: 16 feature maps
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        # Pooling layer (Reduce image size).
        self.pool = nn.MaxPool2d(2)
        ## Second convolution layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        #-------------------------------------------------
        # Decoder 
    
        ## Upsampling layer
        # ConvTranspose2d increases image size.
    
        self.up = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)

        ## Final segmentation layer
        # 1 channel output for tumour vs no tumour
        self.final = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

        # Activation function
        self.relu = nn.ReLU()

    # FORWARD PASS
    def forward(self, x):
        ## First convolution
        x = self.conv1(x)
        # Activation function
        x = self.relu(x)
        # Reduce image size
        x = self.pool(x)

        ## Second convolution
        x = self.conv2(x)
        x = self.relu(x)
    
        ## Upsample image
        x = self.up(x)
        # The final prediction layer
        x = self.final(x)
        # Return segmentation prediction
        return x
