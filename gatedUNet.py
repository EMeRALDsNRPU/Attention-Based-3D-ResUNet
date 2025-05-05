import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# 3D Residual Block
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels,stride,padding1,padding2):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,stride=stride, padding=padding1,dilation=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=padding2,dilation=2)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # If input and output channels are different, use a 1x1 convolution for skip connection
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1,stride=stride) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

# 3D Attention Gate
class AttentionGate3D(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels,padg,padx,padp,padt,padt2,padc,padc2):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Conv3d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=padg,dilation=2, bias=True)
        self.W_x = nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=padx,dilation=2, bias=True)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=padp,dilation=2, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tconv3d = nn.ConvTranspose3d(inter_channels, inter_channels,  kernel_size=1, stride=2, padding=padt)
        self.tconv3d2 = nn.ConvTranspose3d(inter_channels, inter_channels,  kernel_size=1, stride=2, padding=padt2)
        self.conv3d1 = nn.Conv3d(inter_channels, inter_channels,  kernel_size=1, stride=1, padding=padc)
        self.conv3d2 = nn.Conv3d(inter_channels, inter_channels,  kernel_size=1, stride=1, padding=padc2)
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        x1 = self.tconv3d(x1)
        x1 = self.conv3d1(x1)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        x = self.tconv3d2(x)
        x = self.conv3d2(x)
        return x * psi

# 3D U-Net with Residual Blocks and Attention
class LungSegmentationNet3D(nn.Module):
    def __init__(self):
        super(LungSegmentationNet3D, self).__init__()
        
        # Encoder (Down-sampling path)
        self.encoder1 = ResidualBlock3D(1, 8,2,0,3)
        self.encoder2 = ResidualBlock3D(8, 16,2,2,2)
        self.encoder3 = ResidualBlock3D(16, 32,2,2,2)
        self.encoder4 = ResidualBlock3D(32, 64,2,2,2)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(64, 128,1,2,2)
        
        # Decoder (Up-sampling path with Attention Gates)
        self.tconvd4 = nn.ConvTranspose3d(128, 64,  kernel_size=3, stride=2, padding=0)
        self.decoder4 = ResidualBlock3D(64, 64,1,2,2)
        self.Conv3d4 = nn.Conv3d(64,64,  kernel_size=1, stride=1, padding=[1,1,1])
        self.att4 = AttentionGate3D(64, 64,64,[1,1,1],[1,1,1],0,[1,1,1],[1,1,1],[1,1,1],[3,3,3])
        self.decoder4b = nn.Conv3d(128,64,  kernel_size=1, stride=1, padding=[1,1,1])
        
        self.tconvd3 = nn.ConvTranspose3d(64, 32,  kernel_size=3, stride=2, padding=0)
        self.decoder3 = ResidualBlock3D(32, 32,1,[2,2,2],[2,2,2])
        self.Conv3d3 = nn.Conv3d(32,32,  kernel_size=1, stride=1, padding=[0,1,1])
        self.att3 = AttentionGate3D(32,32,32,[0,1,1],0,0,0,0,[7,7,7],[7,7,7])
        self.decoder3b = nn.Conv3d(64,32,  kernel_size=1, stride=1, padding=[0,1,1])
        
        self.tconvd2 = nn.ConvTranspose3d(32, 16,  kernel_size=3, stride=2, padding=0)
        self.decoder2 = ResidualBlock3D(16,16,1,2,2)
        self.Conv3d2 = nn.Conv3d(16,16,  kernel_size=1, stride=1, padding=[0,1,1])
        self.att2 = AttentionGate3D(16, 16, 16,[0,1,1],0,0,0,0,[14,18,18],[14,18,18])
        self.decoder2b =  nn.Conv3d(32,16,  kernel_size=1, stride=1, padding=[0,1,1])
        
        self.tconvd1 = nn.ConvTranspose3d(16, 8,  kernel_size=3, stride=2, padding=0)
        self.decoder1 = ResidualBlock3D(8, 8,1,2,2)
        self.Conv3d1 = nn.Conv3d(8,8,  kernel_size=1, stride=1, padding=[0,1,1])
        self.att1 = AttentionGate3D(8, 8, 8,[0,1,1],0,0,0,0,[28,39,39],[28,39,39])
        self.decoder1b = nn.Conv3d(16,8,  kernel_size=1, stride=1, padding=[0,1,1])
        
        # Final output layer
        self.final_conv = nn.Conv3d(8, 1, stride=1, kernel_size=3)
        self.final_conv2 = nn.Conv3d(1, 1, stride=1, kernel_size=3)
        self.final_conv3 = nn.Conv3d(1, 1, stride=1, kernel_size=3, padding=2)
    def forward(self, x):
        # Encoding path
        print(x.shape)
        e1 = self.encoder1(x)
        print(e1.shape)
        e2 = self.encoder2(e1)
        print(e2.shape)
        e3 = self.encoder3(e2)
        print(e3.shape)
        e4 = self.encoder4(e3)
        print(e4.shape)
        
        # Bottleneck
        b = self.bottleneck(e4)
        print(b.shape)
        # Decoding path with Attention
        
        d4 = self.tconvd4(b)
        print(d4.shape)
        # d4 = F.interpolate(b, scale_factor=2, mode='trilinear', align_corners=True)
        d4 = self.decoder4(d4)
        d4c = self.Conv3d4(d4)
        print(d4.shape)
        d4 = torch.cat((self.att4(e4, d4), d4c), dim=1)
        print(d4.shape)
        d4 = self.decoder4b(d4)
        print(d4.shape)


        d3 = self.tconvd3(d4)
        print(d3.shape)
        d3 = self.decoder3(d3)
        d3c = self.Conv3d3(d3)
        print(d3.shape)
        #d3 = F.interpolate(d4, scale_factor=2, mode='trilinear', align_corners=True)
        d3 = torch.cat((self.att3(e3, d3), d3c), dim=1)
        print(d3.shape)
        d3 = self.decoder3b(d3)
        print(d3.shape)
        

        d2 = self.tconvd2(d3)
        print(d2.shape)
        d2 = self.decoder2(d2)
        d2c = self.Conv3d2(d2)
        print(d2.shape)
        #d2 = F.interpolate(d3, scale_factor=2, mode='trilinear', align_corners=True)
        d2 = torch.cat((self.att2(e2, d2), d2c), dim=1)
        d2 = self.decoder2b(d2)
        print(d2.shape)
        

        d1 = self.tconvd1(d2)
        print(d1.shape)
        d1 = self.decoder1(d1)
        d1c =self.Conv3d1(d1)
        print(d1.shape)
        #d1 = F.interpolate(d2, scale_factor=2, mode='trilinear', align_corners=True)
        d1 = torch.cat((self.att1(e1, d1), d1c), dim=1)
        d1 = self.decoder1b(d1)
        print(d1.shape)
        
        # Final output
        out = self.final_conv(d1)
        # print(out.shape)
        top = int(out.shape[3]/2)-150
        left = top
        imagetc = out
        imagec = torchvision.transforms.functional.crop(imagetc,top, left, 300, 300) 
        imagec = imagec[:,:,int(77/2-23/2):int(77/2+23/2),:,:]
        out =  imagec
        print(out.shape)
        out = self.final_conv2(out)
        print(out.shape)
        out = self.final_conv3(out)
        print(out.shape)
        return torch.sigmoid(out)

# Example usage of the 3D lung segmentation network
if __name__ == '__main__':
    model = LungSegmentationNet3D()
    # Example input: batch_size=1, channels=1 (grayscale), depth=64, height=128, width=128
    input_volume = torch.rand((1, 1, 64, 128, 128))  # Example 3D image volume
    output = model(input_volume)
    print(output.shape)  # Output shape: (1, 1, 64, 128, 128)
