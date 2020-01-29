import cv2
import numpy
import os

def main():
	_path = '/home/mancmanomyst/img_sr/91-image'
	names = os.listdir('/home/mancmanomyst/img_sr/91-image')
	names = sorted(names)
	nums = names.__len__()

	data = []
	label = []

	for i in range(1):
		name = _path + '/'+names[i]
		print(name)
		print('/home/mancmanomyst/img_sr/91-image/t2.bmp')
		#hr_img = cv2.imread('/home/mancmanomyst/img_sr/91-image/t2.bmp', cv2.IMREAD_COLOR)
		#hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
		#hr_img = hr_img[:, :, 0]
		#shape = hr_img.shape
		#print(shape)
	

if __name__ == '__main__':
	main()
