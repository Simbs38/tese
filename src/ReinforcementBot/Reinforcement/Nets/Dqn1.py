import torch.nn as nn
import torch.nn.functional as F
import torch

class Dqn1(nn.Module):
	def __init__(self, utils):
		super(Dqn, self).__init__()
		self.fc1 = nn.Linear(in_features = utils.InputSize, out_features = 800)
		self.fc2 = nn.Linear(in_features = 800, out_features = 400)
		self.out = nn.Linear(in_features = 400, out_features = utils.OutputSize)

	def forward(self, t):
		t = F.relu(self.fc1(t))
		t = F.relu(self.fc2(t))
		t = self.out(t)
		return t