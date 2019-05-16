import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from torchvision import models
from torch.autograd import Variable
from torchvision.models.resnet import model_urls, BasicBlock, ResNet 


class TaskNet(nn.Module):

    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    "Basic class which does classification."
    def __init__(self, num_cls=10, weights_init=None):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls
        self.setup_net()
        self.criterion = nn.CrossEntropyLoss()
        if weights_init is not None:
            self.load(weights_init)
        else:
            init_weights(self)

    def forward(self, x, with_ft=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        score = self.classifier(x)
        if with_ft:
            return score, x
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

class DTNClassifier(TaskNet):
    "Classifier used for SVHN source experiment"

    num_channels = 3
    image_size = 32
    name = 'DTN'
    out_dim = 512 # dim of last feature layer

    def setup_net(self):
        self.conv_params = nn.Sequential (
                nn.Conv2d(self.num_channels, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
    
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512),
                )

        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, self.num_cls)
                )

class SilenceLayer(torch.autograd.Function):
  def __init__(self):
    pass
  def forward(self, input):
    return input * 1.0

  def backward(self, gradOutput):
    return 0 * gradOutput

def init_weights(obj):
    for m in obj.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()

# convnet without the last layer
class DTN(nn.Module):
    def __init__(self, weights_init=None):
        super(DTN, self).__init__()
        self.setup_net()
        self.criterion = nn.CrossEntropyLoss()
        self.__in_features = 512
        if weights_init is not None:
            self.load(weights_init)
        else:
            init_weights(self)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        return x
        
    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def setup_net(self):
        self.conv_params = nn.Sequential (
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout()
                )

    def output_num(self):
        return self.__in_features

class ResNet18Fc(nn.Module):
  def __init__(self):
    super(ResNet18Fc, self).__init__()
    model_resnet18 = models.resnet18(pretrained=False)
    self.conv1 = model_resnet18.conv1
    self.bn1 = model_resnet18.bn1
    self.relu = model_resnet18.relu
    self.maxpool = model_resnet18.maxpool
    self.layer1 = model_resnet18.layer1
    self.layer2 = model_resnet18.layer2
    self.layer3 = model_resnet18.layer3
    self.layer4 = model_resnet18.layer4
    self.avgpool = model_resnet18.avgpool
    self.__in_features = model_resnet18.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class ResNet18_full(ResNet):

    num_channels = 3
    image_size = 224
    name = 'ResNet18'
    out_dim = 512 # dim of last feature layer

    def __init__(self, num_cls=6, weights_init=None):
        super(ResNet18_full, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_cls)
        self.criterion = nn.CrossEntropyLoss()

        if weights_init is not None:
            self.load(weights_init)

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

network_dict = {"DTN":DTN, "ResNet18":ResNet18Fc, "DTN_full": DTNClassifier, "ResNet18_full": ResNet18_full}
