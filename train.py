import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from torch.autograd import Variable

import time

from opt.trainOption import Options
from utils import util
from model import GAN


if __name__ == '__main__':
	opt = Options().parse()
	vis = util.VisdomServer()
	dataset = datasets.MNIST('./dataset/mnist', True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))]), download=True)
	dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

	model = GAN.basicGAN()
	model.initialize(opt)

	for epoch in range(opt.epochs):
		for i, (imgs, _) in enumerate(dataloader, 1):
			#print(imgs.size())
			imgs_num = imgs.size(0)
			real_imgs = Variable(imgs)
			noise = Variable(torch.randn(imgs_num, opt.fake_dim))
			model.update_optimzer(real_imgs, noise)

			img_ret = model.get_current_vis()
			loss_ret = model.get_current_loss()

			if (i + 1) % 100 == 0:
				vis.print_loss(epoch, i, loss_ret)
				vis.plot_loss(epoch, float(i / len(dataloader)), loss_ret)
				vis.show_perdiction(img_ret['fake'][:10])

				model.save('latest.pth')
		model.save(str(epoch) + '.pth')