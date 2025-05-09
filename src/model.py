import torch.nn as nn

class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super(CharCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 28x28 → 14x14

            nn.Conv2d(32, 64, 3, padding=1),  # 14x14 → 14x14
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 14x14 → 7x7

            nn.Conv2d(64, 128, 3, padding=1),  # 7x7
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)
