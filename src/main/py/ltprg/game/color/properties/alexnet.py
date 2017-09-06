import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

class PartialAlexnet:
	def __init__(self, stop_layer):
		# returns trained model up to stop_layer
		# (fc6 or fc7)
		assert stop_layer in ['fc6', 'fc7']
		self.stop_layer = stop_layer
		self.model = models.alexnet(pretrained=True)
		if self.stop_layer == 'fc6':
			x = -5
		elif self.stop_layer == 'fc7':
			x = -2
		self.model.classifier = nn.Sequential(*list(
							self.model.classifier.children()
							)[:x])

	def forward(self, x):
		self.model.eval()
		return self.model.forward(x)

def image_to_alexnet_input(image):
	# preprocess using ImageNet train set statistics
	preprocess_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
									     std=[0.229, 0.224, 0.225])
	return Variable(preprocess_fn(image))

def rgb_to_alexnet_input(rgb):
	# reformat to 1 x 3 x im_width x im_height tensor
	im_width = 224 # required by Alexnet
	im_height = 224
	red = torch.FloatTensor([rgb[0]]).expand(1, im_width, im_height)
	green = torch.FloatTensor([rgb[1]]).expand(1, im_width, im_height)
	blue = torch.FloatTensor([rgb[2]]).expand(1, im_width, im_height)
	image = torch.cat((torch.cat((red, green), 0),
						 blue), 0).unsqueeze(0)
	input_var = image_to_alexnet_input(image)
	return input_var