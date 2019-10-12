import torch.nn as nn
import torch.nn.functional as F
import torch

class Dqn3(nn.Module):
	def __init__(self, utils):
		super(Dqn3, self).__init__()
		self.sigmoid = nn.Sigmoid()
		self.fc1 = nn.Linear(utils.InputSize, 800)
		self.fc2 = nn.Linear(800, 400)
		self.fc3 = nn.Linear(400, 200)
		self.fc4 = nn.Linear(200, 100)
		self.fc5 = nn.Linear(100, utils.OutputSize)
		
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
		return t