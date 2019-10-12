import torch.nn as nn
import torch.nn.functional as F
import torch

class Dqn4(nn.Module):
	def __init__(self, utils):
		super(Dqn4, self).__init__()
		self.sigmoid = nn.Sigmoid()
		self.fc1 = nn.Linear(utils.InputSize, 900)
		self.fc2 = nn.Linear(900, 600)
		self.fc3 = nn.Linear(600, 300)
		self.fc4 = nn.Linear(300, 180)
		self.fc5 = nn.Linear(180, 60)
		self.fc6 = nn.Linear(60, utils.OutputSize)

	def forward(self, t):
		t = self.fc1(t)
		t = self.fc2(t)
		t = self.sigmoid(t)
		t = self.fc3(t)
		t = self.fc4(t)
		t = self.sigmoid(t)
		t = self.fc5(t)
		t = self.sigmoid(t)
		t = self.fc6(t)
		return t