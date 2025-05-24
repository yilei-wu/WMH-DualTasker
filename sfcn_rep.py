import torch
import torch.nn as nn
import torch.nn.functional as F
# from fds import FDS 

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class SFCN_rep(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], mode=1, use_t1=False, use_fds=False):
        super(SFCN_rep, self).__init__()

        if use_t1:
            self.layer1 = self.conv_layer(2, 32, maxpool=True, kernel_size=3, padding=1)
        else:
            self.layer1 = self.conv_layer(1, 32, maxpool=True, kernel_size=3, padding=1)
        if mode == 1:
            self.layer2 = nn.Sequential(
                    nn.Conv3d(32, 64, padding=1, kernel_size=3),
                    nn.BatchNorm3d(64),
                    nn.MaxPool3d((2,2,1), stride=(2,2,1)),
                    nn.ReLU(),
                )
            self.layer3 = self.conv_layer(64, 128, maxpool=False, kernel_size=3, padding=1)
        else:
            self.layer2 = self.conv_layer(32, 64, maxpool=True, kernel_size=3, padding=1)
            self.layer3 = nn.Sequential(
                    nn.Conv3d(64, 128, padding=1, kernel_size=3),
                    nn.BatchNorm3d(128),
                    nn.MaxPool3d((2,2,1), stride=(2,2,1)),
                    nn.ReLU(),
                )
        self.layer4 = self.conv_layer(128, 256, maxpool=False, kernel_size=3, padding=1)
        self.layer5 = self.conv_layer(256, 256, maxpool=False, kernel_size=3, padding=1)
        
        self.layer6 = nn.Conv3d(256, 1, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        # config = dict(feature_dim=[64, 64, 32], start_update=0, start_smooth=1, kernel='gaussian', ks=5, sigma=2)
        # self.FDS = FDS(config)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        cam = self.layer6(x)
        # if self.use_fds:
            # not implemented yet, need to pass the label and epoch to the model 
            # smoothed_cam = self.FDS.smooth(cam, ) 
            # pass
        x = self.gap(cam)

        return x, cam

if __name__ == '__main__':
    temp_input = torch.rand((1, 1, 256, 256, 64))
    model = SFCN_rep(mode=1)
    y1, y2 = model(temp_input)

    print(y1.size())
    print(y2.size())

    model = SFCN_rep(mode=2)
    y1, y2 = model(temp_input)

    print(y1.size())
    print(y2.size())