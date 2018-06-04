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


idx = ['0','1','2','3','4','5','6','7','8','9',
       'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def pil_loader(path):
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('L')



class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1,32, kernel_size=5)
		self.conv2 = nn.Conv2d(32,128, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(3200, 2048)
		self.fc3 = nn.Linear(2048, 512)
		self.fc5 = nn.Linear(512, 47)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x),2))
		x =F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 3200)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.fc3(x))
		x = F.dropout(x, training=self.training)
		x = self.fc5(x)
		return x



def train(epoch, model, optimizer, loader):
	model.train()
	for batch_idx, (data, target) in enumerate(loader):
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = nn.CrossEntropyLoss()
		output = loss(output, target)
		output.backward()
		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch, batch_idx * len(data), len(loader.dataset),
                 100. * batch_idx / len(loader), output.data.item()))



def predict_char(gray, model):
    w = gray.size[0]
    h = gray.size[1]
    gray = gray.convert('L')
    gray = gray.point(lambda x: 0 if x<180 else 255, '1')
    x= int(16- (w/2))
    y = int(16- (h/2))
    canvas = Image.new('L', (32, 32), (255))
    canvas.paste(gray, box=(x, y))

    #canvas = ImageOps.invert(canvas)
    canvas = np.array(canvas)
    #canvas = canvas / 255.0
    
    #test_data = np.array(gray)
    test_output = model(Variable(torch.FloatTensor(canvas).unsqueeze(0).unsqueeze(0).data))
    #print (test_output.data.max(1, keepdim=True))
    pred = test_output.data.max(1, keepdim=True)[1] 
    pred = np.array(pred).squeeze(0).squeeze(0)
    #print('El caracter es =>', idx[pred])
    #plt.imshow(canvas)
    #plt.show()
    return idx[pred]

'''if __name__ == "__main__":
	#trans = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
	#mydata = dset.ImageFolder('./char74k',transform=trans, loader=pil_loader)
	#loader = torch.utils.data.DataLoader(mydata, batch_size=128, shuffle=True, num_workers=2)
	
	#model = Net()
	#optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-4)
	for epoch in range(5):
	train(epoch)

	model = Net()
	model.load_state_dict(torch.load("char_recognizer.pt"))
	res = ""
	for i in range(6):
		pil_im = Image.open("./Test/test{}.jpg".format(i))
		res += predict_char(pil_im)
	print ('La placa es:  ', res)

	#torch.save(model.state_dict(), 'char_recognizer.pt')
'''
