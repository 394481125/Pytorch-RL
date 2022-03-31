import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
  def __init__(self,num_frame,num_action):
    super(DQN,self).__init__()
    self.num_frame = num_frame
    self.num_action = num_action
    self.conv1 = nn.Conv2d(in_channels=self.num_frame,out_channels=16,kernel_size=8,stride=4,padding=2)
    self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2)
    self.fc1 = nn.Linear(32 * 81,256)
    self.fc2 = nn.Linear(256,self.num_action)
  def forward(self,image):
    x = F.relu(self.conv1(image))
    x = F.relu(self.conv2(x))
    x = x.view(-1,32*81)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x