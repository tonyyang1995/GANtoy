import torch
import torch.nn as nn
from model import network
import torch.optim as optim
import os

class basicGAN(nn.Module):
	def __init__(self):
		super(basicGAN, self).__init__()

	def name(self):
		return 'basicGAN'

	def set_requires_grad(self, nets, requires_grad=False):
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad

	def initialize(self, opt):
		self.loss_name = ['GAN']
		self.checkpoint = opt.checkpoint
		self.name = opt.name

		self.netG = network.define_G('basic', opt)
		self.netD = network.define_D('basic')
		
		if opt.loadmodel:
			self.load(opt.load_path)

		self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.lr)
		self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr)

		self.GANcriterion = network.GANLoss()


	def forward(self, real_img, noise):
		self.real_img = real_img
		self.fake_img = self.netG(noise)
		
	def backward_D(self):
		# fake
		fake_out = self.netD(self.fake_img.detach())
		self.loss_D_fake = self.GANcriterion(fake_out, False)

		# real
		real_out = self.netD(self.real_img)
		self.loss_D_real = self.GANcriterion(real_out, True)

		# combined Loss
		self.loss_D = (self.loss_D_fake + self.loss_D_real) #* .5

		self.loss_D.backward()

	def backward_G(self):
		fake_out = self.netD(self.fake_img)
		self.loss_G = self.GANcriterion(fake_out, True)
		self.loss_G.backward()

	def update_optimzer(self, real_img, noise):
		self.forward(real_img, noise)

		# update D
		self.set_requires_grad(self.netD, True)
		self.optimizer_D.zero_grad()
		self.backward_D()
		self.optimizer_D.step()
		self.set_requires_grad(self.netD, False)

		# update G
		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()


	def get_current_vis(self):
		from collections import OrderedDict
		visual_ret = OrderedDict()
		visual_ret['fake'] = self.fake_img
		visual_ret['real'] = self.real_img
		return visual_ret

	def get_current_loss(self):
		#self.error_cnt += 1
		from collections import OrderedDict
		error_ret = OrderedDict()
		error_ret['Loss_D'] = self.loss_D
		error_ret['Loss_G'] = self.loss_G
		error_ret['D_real'] = self.loss_D_real
		error_ret['D_fake'] = self.loss_D_fake

		return error_ret

	def save(self, epoch_name):
		Gpath = os.path.join(self.checkpoint, self.name, 'G_' + epoch_name)
		Dpath = os.path.join(self.checkpoint, self.name, 'D_' + epoch_name)
		torch.save(self.netG.state_dict(), Gpath)
		torch.save(self.netD.state_dict(), Dpath)

	def load(self, epoch_name):
		Gpath = os.path.join(self.checkpoint, self.name, 'G_' + epoch_name)
		Dpath = os.path.join(self.checkpoint, self.name, 'D_' + epoch_name)
		self.netG.load_state_dict(torch.load(Gpath))
		self.netD.load_state_dict(torch.load(Dpath))
