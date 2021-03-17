import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):

    """
    Class EmbeddingNet
    First module of Relation Network
    Takes input tensor of (way * shot, 3, 28, 28) size and returns output tensor of (way * shot, 64, 5, 5) size.

    """
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.block1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.block4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        return out4



class RelationNet(nn.Module):

    """
    Class RelationNet
    Second module of Relation Network.
    Takes (way * batch, 128, 5, 5) tensor as input and returns (way * batch, 1) probabilities.

    """
    def __init__(self):
        super(RelationNet, self).__init__()
        self.block1 = nn.Sequential(
                        nn.Conv2d(64*2,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8,1)

    def forward(self,x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = out2.view(out2.size(0),-1)
        out4 = F.relu(self.fc1(out3))
        out5 = F.sigmoid(self.fc2(out4))
        return out5
