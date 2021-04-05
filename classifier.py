import torch as tc
import numpy as np


class ResBlock(tc.nn.Module):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.conv_sequence = tc.nn.Sequential(
            tc.nn.Conv2d(filters, filters, (3,3), stride=(1,1), padding=(1,1)),
            tc.nn.BatchNorm2d(filters),
            tc.nn.ReLU(),
            tc.nn.Conv2d(filters, filters, (3,3), stride=(1,1), padding=(1,1)),
            tc.nn.BatchNorm2d(filters)
        )

    def forward(self, x):
        return tc.nn.ReLU()(x + self.conv_sequence(x))


class DownsamplingConvBlock(tc.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownsamplingConvBlock, self).__init__()
        self.stack = tc.nn.Sequential(
            tc.nn.Conv2d(input_channels, output_channels, (4,4), stride=(2,2), padding=(1,1)),
            tc.nn.ReLU(),
            ResBlock(output_channels)
        )

    def forward(self, x):
        return self.stack(x)


class SmallConvNetClassifier(tc.nn.Module):
    def __init__(self, img_height, img_width, img_channels, num_filters, num_classes):
        super(SmallConvNetClassifier, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.conv_stack = tc.nn.Sequential(
            DownsamplingConvBlock(img_channels, num_filters),
            DownsamplingConvBlock(num_filters, num_filters),
            DownsamplingConvBlock(num_filters, num_filters)
        )
        self.num_conv_features = (img_height // 8) * (img_width // 8) * num_filters
        self.fc = tc.nn.Linear(self.num_conv_features, self.num_classes)

    def forward(self, x):
        spatial_features = self.conv_stack(x)
        flat_features = tc.reshape(spatial_features, (-1, self.num_conv_features))
        logits = self.fc(flat_features)
        return logits
