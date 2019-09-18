import torch.nn as nn
import torch.nn.functional as F
import torch

class Dqn2(nn.Module):
	def __init__(self, utils):
		super(Dqn2, self).__init__()
		self.sigmoid = nn.Sigmoid()
		self.fc1 = nn.Linear(utils.InputSize, 400)
		self.fc2 = nn.Linear(400, 100)
		self.fc3 = nn.Linear(100, utils.OutputSize)
		
	def forward(self, t):
		t = self.fc1(t)
		t = self.sigmoid(t)
		t = self.fc2(t)
		t = self.sigmoid(t)
		t = self.fc3(t)
		return t