import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

def init_weights(net, init_type='xavier', gain=0.2):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init_normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			else:
				raise NotImplementedError('initialization method is not implemented')
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)

		print('initialize network')
		net.apply(init_func)

def init_net(net, init_type='xavier', gpu_ids=[]):
	# apply gpu_ids here

	init_weights(net, init_type)
	return net

# def get_scheduler(optimizer, lr_policy, opt):
# 	# if lr_policy == 'lambda':
# 	# 	def lambda_rule(epoch):
# 	# 		lr_l = 1.0 - max(0, (epoch + 1 + opt.epochs) / float(opt.epoch_decay + 1))
# 	# 		return lr_l
# 	# 	scheduler = lr_scheduler
# 	if lr_policy == 'step':
# 		scheduler = lr_scheduler.StepLR(optimizer, step_size)

def define_G(net_type, opt):
	if net_type == 'basic':
		netG = baseGenerator(opt)
	else:
		raise NotImplementedError("net_type not found")

	return init_net(netG)

def define_D(net_type):
	if net_type == 'basic':
		netD = baseDiscriminator()
	else:
		raise NotImplementedError("net_type not found")

	return init_net(netD)

class baseDiscriminator(nn.Module):
	def __init__(self):
		super(baseDiscriminator, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1,32,5,1,2), 
			nn.LeakyReLU(.2, True),
			nn.MaxPool2d(2)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(32,64,5,1,2),
			nn.LeakyReLU(.2, True),
			nn.MaxPool2d(2)
		)

		self.fc = nn.Sequential(
			nn.Linear(64 * 7 * 7, 1024),
			nn.LeakyReLU(.2, True),
			nn.Linear(1024, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x.squeeze()


class baseGenerator(nn.Module):
	def __init__(self, opt):
		super(baseGenerator, self).__init__()
		self.fc = nn.Linear(opt.fake_dim, 1*56*56)
		self.br = nn.Sequential(
			nn.BatchNorm2d(1),
			nn.ReLU(True)
		)

		self.deconv1 = nn.Sequential(
			nn.Conv2d(1,50, 3,1,1),
			nn.BatchNorm2d(50),
			nn.ReLU(True)
		)

		self.deconv2 = nn.Sequential(
			nn.Conv2d(50, 25, 3,1,1),
			nn.BatchNorm2d(25),
			nn.ReLU(True)
		)

		self.deconv3 = nn.Sequential(
			nn.Conv2d(25, 1, 2,2),
			nn.Tanh()
		)
	def forward(self, x):
		x = self.fc(x)
		x = self.br(x.view(x.size(0),1,56,56))
		x = self.deconv1(x)
		x = self.deconv2(x)
		x = self.deconv3(x)
		return x

class GANLoss(nn.Module):
	def __init__(self, target_real=1.0, target_fake=0.0):
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real))
		self.register_buffer('fake_label', torch.tensor(target_fake))
		self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(input)

	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)
