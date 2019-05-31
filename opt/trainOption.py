import argparse
import os
import torch
from utils import util

class Options():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		parser.add_argument('--batch_size', type=int, default=10)
		parser.add_argument('--lr', type=float, default=2e-4)
		parser.add_argument('--checkpoint', type=str, default='./checkpoint')
		parser.add_argument('--name', type=str, default='oriGAN')
		parser.add_argument('--isTrain', type=bool, default=True)

		parser.add_argument('--epochs', type=int, default=10)
		parser.add_argument('--fake_dim', type=int, default=100)

		parser.add_argument('--loadmodel', action='store_true')
		parser.add_argument('--load_path', type=str, default='latest.pth')

		self.initialized = True
		return parser

	def parse(self):
		if not self.initialized:
			parser = argparse.ArgumentParser(
				formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		# if use GPU, set GPU id here

		opt = parser.parse_args()
		self.print_options(opt)
		return opt


	def print_options(self, opt):
		log = ''
		log += '------------- Options -------------\n'
		for k, v in sorted(vars(opt).items()):
			log += '{:>25}: {:<30}\n'.format(str(k), str(v))

		log += '---------------- End ---------------'
		print(log)

		# save to disk
		dirs = os.path.join(opt.checkpoint, opt.name)
		util.mkdirs(dirs)
		fname = os.path.join(dirs, 'opt.txt')
		with open(fname, 'wt') as opt_f:
			opt_f.write(log + '\n')

