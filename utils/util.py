import os
import visdom
import numpy as np

def mkdirs(paths):
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			if not os.path.exists(path):
				os.makedirs(path)
	else:
		if not os.path.exists(paths):
			os.makedirs(paths)

class VisdomServer():
	def __init__(self):
		self.viz = visdom.Visdom()

	def show_perdiction(self, img_tensors):
		if self.viz.win_exists:
			self.viz.close
		self.viz.images(img_tensors, win=2)

	def plot_loss(self, epoch, iter_ratio, loss):
		if not hasattr(self, 'plot_data'):
			self.plot_data = {'X': [], 'Y': [], 'legend': list(loss.keys())}

		if self.viz.win_exists:
			self.viz.close

		self.plot_data['X'].append(epoch + iter_ratio)
		self.plot_data['Y'].append([loss[k].cpu().detach().numpy() for k in self.plot_data['legend']])

		self.viz.line(
			X = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1), 
			Y = np.array(self.plot_data['Y']),
			opts = {
			'title': 'loss over time',
			'legend': self.plot_data['legend'],
			'xlabel': 'epoch',
			'ylabel': 'loss'
			}, win=1)

	def print_loss(self, epoch, iter, loss):
		log = '(epochs: %d, iters: %d)' % (epoch, iter)
		for k, v in loss.items():
			log += '%s: %.3f, ' % (k, v)

		print(log)
		with open('log.txt', 'a') as log_file:
			log_file.write('%s\n' % log)
