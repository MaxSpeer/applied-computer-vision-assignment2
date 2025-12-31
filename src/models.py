import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

class Net(nn.Module):
    def __init__(self, in_ch, num_positions, embedder="normal"):
        super().__init__()
        kernel_size = 3

        # If the embedder should be with strided convolutions, 
        if embedder == "strided":
            stride = 2
            self.pool = lambda x: x # identity, since strided conv reduces size
        else:
            stride = 1
            self.pool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(in_ch, 25, kernel_size, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(25, 50, kernel_size, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(50, 100, kernel_size, padding=1, stride=stride)


        self.fc1 = nn.Linear(100 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, num_positions)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class LateFusionModel(nn.Module):
    def __init__(self, embedder = "normal"):
        super().__init__()
        self.rgb_net = Net(4, 1, embedder=embedder).to(device)
        self.xyz_net = Net(4, 1, embedder=embedder).to(device)
        # TODO: pretrain models and freeze
        #for param in self.rgb_net.parameters():
        #    param.requires_grad = False
        #for param in self.xyz_net.parameters():
        #    param.requires_grad = False

        num_positions = 1 # classifcation cube or sphere
        self.fc1 = nn.Linear(num_positions * 2, num_positions * 10)
        self.fc2 = nn.Linear(num_positions * 10, num_positions)

    def forward(self, x_img, x_xyz):
        x_rgb = self.rgb_net(x_img)
        x_xyz = self.xyz_net(x_xyz)
        x = torch.cat((x_rgb, x_xyz), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class IntermediateFusionNet(nn.Module):
    def __init__(self, rgb_ch, xyz_ch, fusion_type="concat", embedder="normal"):


        # If the embedder should be with strided convolutions, 
        if embedder == "strided":
            stride = 2
            self.pool = lambda x: x # identity, since strided conv reduces size
        else:
            stride = 1
            self.pool = nn.MaxPool2d(2)

        kernel_size = 3
        num_positions = 1
        super().__init__()
        self.fusion_type = fusion_type
        self.rgb_conv1 = nn.Conv2d(rgb_ch, 25, kernel_size, padding=1, stride=stride)
        self.rgb_conv2 = nn.Conv2d(25, 50, kernel_size, padding=1, stride=stride)
        self.rgb_conv3 = nn.Conv2d(50, 100, kernel_size, padding=1, stride=stride)

        self.xyz_conv1 = nn.Conv2d(xyz_ch, 25, kernel_size, padding=1, stride=stride)
        self.xyz_conv2 = nn.Conv2d(25, 50, kernel_size, padding=1, stride=stride)
        self.xyz_conv3 = nn.Conv2d(50, 100, kernel_size, padding=1, stride=stride)

        if fusion_type == "concat":
            self.fc1 = nn.Linear(200 * 8 * 8, 1000)
        else:
            self.fc1 = nn.Linear(100 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, num_positions)

    def forward(self, x_rgb, x_xyz):
        x_rgb = self.pool(F.relu(self.rgb_conv1(x_rgb)))
        x_rgb = self.pool(F.relu(self.rgb_conv2(x_rgb)))
        x_rgb = self.pool(F.relu(self.rgb_conv3(x_rgb)))

        x_xyz = self.pool(F.relu(self.xyz_conv1(x_xyz)))
        x_xyz = self.pool(F.relu(self.xyz_conv2(x_xyz)))
        x_xyz = self.pool(F.relu(self.xyz_conv3(x_xyz)))

        # switch
        if self.fusion_type == "concat":
            x = torch.cat((x_rgb, x_xyz), 1)
        elif self.fusion_type == "add":
            x = x_rgb + x_xyz
        elif self.fusion_type == "hadamard":
            x = x_rgb * x_xyz
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x