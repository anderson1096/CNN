import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from PIL import ImageOps
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import time
import cnn 

if __name__ == "__main__":
	trans = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
	mydata = dset.ImageFolder('./char74k',transform=trans, loader=cnn.pil_loader)
	loader = torch.utils.data.DataLoader(mydata, batch_size=128, shuffle=True, num_workers=2)

	model = cnn.Net()
	optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-4)
	for epoch in range(5):
		cnn.train(epoch, model, optimizer, loader)
	torch.save(model.state_dict(), 'char_recognizer.pt')
