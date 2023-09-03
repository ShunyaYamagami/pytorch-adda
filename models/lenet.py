"""LeNet model for ADDA."""

import torch.nn.functional as F
from torch import nn


class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self, img_size):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            # nn.Conv2d(1, 20, kernel_size=5),
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        # self.fc1 = nn.Linear(50 * 4 * 4, 500)
        # self.fc1 = nn.Linear(50 * 61 * 61, 500)

        if img_size ==64:
            dim = 8450
        elif img_size == 255:
            dim = 180000
        else:
            raise ValueError(f'img_size: {img_size}')
        self.fc1 = nn.Linear(dim, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        # feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        conv_out = conv_out.view(conv_out.size(0), -1)
        feat = self.fc1(conv_out)
        return feat


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self,num_classes):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        # self.fc2 = nn.Linear(500, 10)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
