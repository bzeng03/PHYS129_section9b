#!/usr/bin/env python

import numpy as np
from torch import flatten, tensor, set_default_device, get_default_device, cuda, float32, save, load, manual_seed, use_deterministic_algorithms
from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, CrossEntropyLoss, Sigmoid, Softmax
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

from my_lib import tqdm
from mnist import train_images, train_labels, test_images, test_labels

if cuda.is_available():
	set_default_device('cuda')

class CNN(Module):
	def __init__(self):
		super().__init__()
		self.conv = Conv2d(in_channels=1, out_channels=3, kernel_size=6)
		self.relu = ReLU()
		self.pool = MaxPool2d(2, padding=1)
		self.fc1 = Linear(3*12*12, 60)
		self.sigmoid = Sigmoid()
		self.fc2 = Linear(60, 10)
		self.softmax = Softmax(1)
	def forward(self, x):
		x = self.pool(self.relu(self.conv(x)))
		x = flatten(x, 1)
		x = self.sigmoid(self.fc1(x))
		x = self.softmax(self.fc2(x))
		return x
	def train_all(self, dataset, batch_size, lr, epochs):
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
		crit = CrossEntropyLoss()
		opt = Adam(self.parameters(), lr=lr)
		losses = []
		with tqdm(total=len(loader)*epochs) as progress:
			for epoch in range(epochs):
				self.train()
				total_loss = 0
				for images, labels in loader:
					opt.zero_grad()
					outputs = self(images)
					loss = crit(outputs, labels)
					loss.backward()
					opt.step()
					total_loss += loss.item()
					progress.update()
				losses.append(total_loss)
		return losses

class MLP(Module):
	def __init__(self):
		super().__init__()
		self.fc1 = Linear(28*28, 700)
		self.relu = ReLU()
		self.fc2 = Linear(700, 500)
		self.sigmoid = Sigmoid()
		self.fc3 = Linear(500, 10)
		self.softmax = Softmax(1)
	def forward(self, x):
		x = flatten(x, 1)
		x = self.relu(self.fc1(x))
		x = self.sigmoid(self.fc2(x))
		x = self.softmax(self.fc3(x))
		return x
	def train_all(self, dataset, batch_size, lr, epochs):
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
		crit = CrossEntropyLoss()
		opt = Adam(self.parameters(), lr=lr)
		losses = []
		with tqdm(total=len(loader)*epochs) as progress:
			for epoch in range(epochs):
				self.train()
				total_loss = 0
				for images, labels in loader:
					opt.zero_grad()
					outputs = self(images)
					loss = crit(outputs, labels)
					loss.backward()
					opt.step()
					total_loss += loss.item()
					progress.update()
				losses.append(total_loss)
		return losses

dataset = TensorDataset(
	tensor(train_images, dtype=float32).unsqueeze(1),
	tensor(train_labels, dtype=float32)
)

def display_confusion(model):
	confusion = np.zeros((10, 10), dtype=int)
	predicted = model.forward(tensor(test_images, dtype=float32).unsqueeze(1)).argmax(1)
	for actual, pred in zip(test_labels, predicted):
		confusion[actual, pred] += 1
	fig, ax = plt.subplots()
	im = ax.imshow(confusion)
	ax.set_xticks(range(10))
	ax.set_yticks(range(10))
	for i in range(10):
		for j in range(10):
			ax.text(j, i, confusion[i, j], ha='center', va='center', color='w')
	ax.set_xlabel('Predicted')
	ax.set_ylabel('Actual')
	plt.show()

cnn = CNN()
losses = cnn.train_all(dataset, 64, 0.001, 128)
plt.plot(losses)
plt.show()

display_confusion(cnn)

mlp = MLP()
losses = mlp.train_all(dataset, 64, 0.001, 128)
plt.plot(losses)
plt.show()

display_confusion(mlp)