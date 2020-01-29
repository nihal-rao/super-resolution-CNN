import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2

from srcnn.model import SRCNN
from srcnn.utils import calc_psnr


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights-file', type=str, required=True)
	parser.add_argument('--image-file', type=str, required=True)
	parser.add_argument('--scale', type=int, default=3)
	args = parser.parse_args()

	cudnn.benchmark = True
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	model = SRCNN().to(device)

	state_dict = model.state_dict()
	for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
		if n in state_dict.keys():
			state_dict[n].copy_(p)
		else:
			raise KeyError(n)

	model.eval()

	img = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
	shape = img.shape

	img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	hr_img = img[:, :, 0]
	
	lr_img = cv2.resize(hr_img, (shape[1] // 3, shape[0] // 3), interpolation = cv2.INTER_CUBIC)
	lr_img = cv2.resize(lr_img, (shape[1], shape[0]), interpolation = cv2.INTER_CUBIC)
	bicubic = np.array([lr_img, img[:,:, 1], img[:,:, 2]]).transpose([1, 2, 0])
	bicubic = cv2.cvtColor(bicubic, cv2.COLOR_YCrCb2BGR)
	cv2.imwrite(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)),bicubic)

	lr_img = np.array(lr_img).astype(np.float32)
	lr_img/=255.
	lr_img = torch.from_numpy(lr_img).to(device)
	lr_img = lr_img.unsqueeze(0).unsqueeze(0)

	with torch.no_grad():
		out = model(lr_img).clamp(0.0, 1.0)

	
	hr_img = np.array(hr_img).astype(np.float32)
	hr_img /= 255.
	psnr = calc_psnr(lr_img, hr_img)
	print('PSNR_bicubic: {:.2f}'.format(psnr))
	hr_img=hr_img[6:-6,6:-6]
	psnr = calc_psnr(out, hr_img)
	print('PSNR_SRCNN: {:.2f}'.format(psnr))

	out = out.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

	img = img[6:-6,6:-6,:]
	output = np.array([img[:,:,0], img[:,:, 1], img[:,:, 2]]).transpose([1, 2, 0])
	print(output.shape)
	output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)
	cv2.imwrite(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)),output)

