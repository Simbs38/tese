import torch.nn as nn
import torch.nn.functional as F
import torch

class Dqn5(nn.Module):
	def __init__(self, utils):
		super(Dqn5, self).__init__()
		self.sigmoid = nn.Sigmoid()
		self.fc1  = nn.Linear(utils.InputSize, 2000)
		self.fc2  = nn.Linear(2000, 2500)
		self.fc3  = nn.Linear(2500, 3000)
		self.fc4  = nn.Linear(3000, 2500)
		self.fc5  = nn.Linear(2500, 2000)
		self.fc6  = nn.Linear(2000, 1500)
		self.fc7  = nn.Linear(1500, 1000)
		self.fc8  = nn.Linear(1000, 500 )
		self.fc9  = nn.Linear(500 , 250 )
		self.fc10 = nn.Linear(250, utils.OutputSize)

	def forward(self, t):
		t = self.fc1(t)
		t = self.sigmoid(t)
		t = self.fc2(t)
		t = self.sigmoid(t)
		t = self.fc3(t)
		t = self.sigmoid(t)
		t = self.fc4(t)
		t = self.sigmoid(t)
		t = self.fc5(t)
		t = self.sigmoid(t)
		t = self.fc6(t)
		t = self.sigmoid(t)
		t = self.fc7(t)
		t = self.sigmoid(t)
		t = self.fc8(t)
		t = self.sigmoid(t)
		t = self.fc9(t)
		t = self.sigmoid(t)
		t = self.fc10(t)
		return t