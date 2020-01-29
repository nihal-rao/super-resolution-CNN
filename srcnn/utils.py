import os
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_PATH = '/home/mancmanomyst/img_sr/91-image' #insert folder path containing training images here
TEST_PATH = '/home/mancmanomyst/img_sr/Set5' #insert folder path containing eval images here
Random_Crop = 30 #no. of crops of each image
Patch_size = 33	#size of input low resolution "sub-images"
label_size = 21 #size of target high resolution images
conv_side = 6 
scale = 3 #scale for producing low resolution images
BLOCK_STEP = 16 #stride while cropping "sub-images" for training
BLOCK_SIZE = 33	#size of input low resolution training "sub-images"


def prepare_train_data(_path):
	"""
	Used to prepare training sub-images and corresponding high resolution targets from images, by taking crops at a specified stride.
	"""
	names = os.listdir(_path)
	names = sorted(names)
	nums = names.__len__()

	data = []
	target = []

	for i in range(nums):
		name = _path + '/' + names[i]
		hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
		hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
		hr_img = hr_img[:, :, 0]
		shape = hr_img.shape

		# two resize operation to produce training data and targets
		lr_img = cv2.resize(hr_img, (shape[1] // scale, shape[0] // scale), interpolation = cv2.INTER_CUBIC)
		lr_img = cv2.resize(lr_img, (shape[1], shape[0]), interpolation = cv2.INTER_CUBIC)

		width_num = (shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
		height_num = (shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) // BLOCK_STEP
		for k in range(width_num):
			for j in range(height_num):
				x = k * BLOCK_STEP
				y = j * BLOCK_STEP
				hr_patch = hr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
				lr_patch = lr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]

				lr_patch = lr_patch.astype(float) / 255.
				hr_patch = hr_patch.astype(float) / 255.

				lr = np.zeros((1, Patch_size, Patch_size), dtype=np.double)
				hr = np.zeros((1, label_size, label_size), dtype=np.double)

				lr[0, :, :] = lr_patch
				hr[0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]

				data.append(lr)
				target.append(hr)

	data = np.array(data, dtype=float)
	target = np.array(target, dtype=float)
	return data, target

def prepare_test_data(_path):
	"""
	Used to prepare test data.
	"""
	names = os.listdir(_path)
	names = sorted(names)
	nums = names.__len__()

	data = np.zeros((nums * Random_Crop, 1, Patch_size, Patch_size), dtype=np.double)
	target = np.zeros((nums * Random_Crop, 1, label_size, label_size), dtype=np.double)

	for i in range(nums):
		name = _path + '/' + names[i]
		hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
		shape = hr_img.shape

		hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
		hr_img = hr_img[:, :, 0]

		# two resize operation to produce training data and target
		lr_img = cv2.resize(hr_img, (shape[1] // scale, shape[0] // scale), interpolation = cv2.INTER_CUBIC)
		lr_img = cv2.resize(lr_img, (shape[1], shape[0]), interpolation = cv2.INTER_CUBIC)

		# produce Random_Crop random coordinate to crop training img
		Points_x = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)
		Points_y = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)

		for j in range(Random_Crop):
			lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
			hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]

			lr_patch = lr_patch.astype(float) / 255.
			hr_patch = hr_patch.astype(float) / 255.

			data[i * Random_Crop + j, 0, :, :] = lr_patch
			target[i * Random_Crop + j, 0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
			# cv2.imshow("lr", lr_patch)
			# cv2.imshow("hr", hr_patch)
			# cv2.waitKey(0)
	return data, target


def write_hdf5(data, labels, output_filename):
	"""
	Used to save low resolution image data and corresponding high resolution target to hdf5 file.
	"""

	x = data.astype(np.float32)
	y = labels.astype(np.float32)

	with h5py.File(output_filename, 'w') as h:
		h.create_dataset('data', data=x, shape=x.shape)
		h.create_dataset('target', data=y, shape=y.shape)
		# h.create_dataset()


def read_training_data(file):
	"""
	Used to read .h5 and extract data, labels from .h5 file
	"""
	with h5py.File(file, 'r') as hf:
		data = np.array(hf.get('data'))
		target = np.array(hf.get('target'))
		return data, target

def calc_psnr(hr_img, gt_img):
	#returns peak signal to noise ratio of the high resolution output and ground truth image.
	return 10. * torch.log10(1. / torch.mean((hr_img - gt_img) ** 2))


class AverageMeter(object):
	"""
	used to store avg psnr.
	"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count



if __name__ == "__main__":
	data, target = prepare_train_data(DATA_PATH)
	write_hdf5(data, target, "crop_train.h5")
	data, target = prepare_test_data(TEST_PATH)
	write_hdf5(data, target, "test.h5")
	# _, _a = read_training_data("train.h5")
	# _, _a = read_training_data("test.h5")