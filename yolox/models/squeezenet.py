import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict

class FloorLayer(torch.autograd.Function):

	@staticmethod
	def forward(self, x):
		out = x.abs().floor() * x.sign()
		return out

	@staticmethod
	def backward(self, grad_output):
		grad_input = grad_output.clone()
		return grad_input

class convlayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, groups=1):
		super(convlayer, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
							  bias=False, groups=groups)
		self.bn = nn.BatchNorm2d(out_channels)
		self.activation = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.conv(x)
		out = self.bn(out)
		out = self.activation(out)
		return out

class Fire3B(nn.Module):
	def __init__(self, inplanes, squeeze_planes,
				 expand1x1_planes, expand3x3_planes):
		super(Fire3B, self).__init__()
		self.inplanes = inplanes
		self.outplanes = expand1x1_planes + expand3x3_planes
		self.squeeze = convlayer(in_channels=inplanes, out_channels=squeeze_planes, kernel_size=1)
		self.expand1x1 = convlayer(in_channels=squeeze_planes, out_channels=expand1x1_planes, kernel_size=1)
		self.expand3x3 = convlayer(in_channels=squeeze_planes, out_channels=expand3x3_planes, kernel_size=3, padding=1)

		self.floor = FloorLayer.apply
	def forward(self, x):
		skip = x
		x = self.squeeze(x)
		out = torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)
		# out = self.se(out)
		if self.inplanes == self.outplanes:
			out = out + skip

		return out

class Fire3(nn.Module):
	def __init__(self, inplanes, squeeze_planes,
				 expand1x1_planes, expand3x3_planes):
		super(Fire3, self).__init__()
		self.inplanes = inplanes
		self.outplanes = expand1x1_planes + expand3x3_planes
		self.squeeze = convlayer(in_channels=inplanes, out_channels=squeeze_planes, kernel_size=1)
		self.expand1x1 = convlayer(in_channels=squeeze_planes, out_channels=expand1x1_planes, kernel_size=1)
		self.expand3x3 = nn.Sequential(convlayer(in_channels=squeeze_planes, out_channels=expand3x3_planes, kernel_size=1, padding=0),
									   convlayer(in_channels=expand3x3_planes, out_channels=expand3x3_planes, kernel_size=3, padding=1, groups=expand3x3_planes),
									   convlayer(in_channels=expand3x3_planes, out_channels=expand3x3_planes,
												 kernel_size=1, padding=0))

		self.floor = FloorLayer.apply
	def forward(self, x):
		skip = x
		x = self.squeeze(x)
		out = torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)
		# out = self.se(out)
		if self.inplanes == self.outplanes:
			out = out + skip

		return out

def SqueezeNet_cw(**kwargs):
	return SqueezeNet_cw_net(**kwargs)


class SqueezeNet_cw_net(nn.Module):

	def __init__(self, used_layers=[-1], extra_conv=False):
		super(SqueezeNet_cw_net, self).__init__()
		self.used_layers = used_layers
		self.feature = nn.Sequential(
			# convlayer(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=2),
			convlayer(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			Fire3(32, 16, 64, 64),
			Fire3(128, 16, 64, 64),
			nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
			Fire3(128, 32, 128, 128),
			Fire3(256, 64, 128, 128))
		self.extra_conv = extra_conv
		if extra_conv:
			self.feature2 = nn.Sequential(OrderedDict([('1', Fire3B(256,48,192,192))]))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_uniform_(m.weight)
				if m.bias is not None:
					init.constant_(m.bias, 0)

	def feature_all(self, input, dict =[16,29]):#[3,7,10,14]
		output = []
		cnt = 0
		for module in self.feature._modules.values():
			#'conv' in str(type(module))
			input = module(input)
			if cnt in dict:
				# print(cnt,input.size())
				output.append(input)
			cnt +=1
		return output

	def forward(self, x):
		if self.used_layers[0]==-1:
			x = self.feature(x)
			if self.extra_conv:
				x = self.feature2(x)
			# print(x.size())
		else:
			x= self.feature_all(x, self.used_layers)
			if self.extra_conv:
				x[-1] = self.feature2(x[-1])
			# print(x.size())
		#c
		return x 