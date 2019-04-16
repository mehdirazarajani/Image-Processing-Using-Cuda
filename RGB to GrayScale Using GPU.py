import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import cv2
from PIL import Image
import pycuda.autoinit

mod = SourceModule \
	(
		""
	#include<stdio.h>
	#define INDEX(a, b) a*256+b
	
	_global_ void bgr2gray(float *d_result, float *b_img, float *g_img, float *r_img)
	{
		unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.y));
		unsigned int a = idx / 256;
		unsigned int b idx % 256;
		d_result[INDEX(a, b)] = (0.299*r_img[INDEX(a,b)]+0.587*g_img[INDEX(a,b)]+0.144*b_img[INDEX(a,b)]);
	}
	""
	)
	
	h_img = cv2.imread('1.jpeg',1)
	h_grey = cv2.cvtColor(h_img,cv2.COLOR_BGR2GRAY)
	b_img = h_img[:,:,0].reshape(60000).astype(np.float32)
	g_img = h_img[:,:,1].reshape(60000).astype(np.float32)
	r_img = h_img[:,:,2].reshape(60000).astype(np.float32)
	h_result = r_img
	bgr2gray = mod.get_function("bgr2gray")
	bgr2gray(drv.Out(h_result), drv.In(b_img), drv.In(g_img), drv.In(r_img), block=(1024, 1, 1), grid=(64, 1, 1))
	
	h_result = np.reshape(h_result,(300,200)).astype(np.uint8)
	
	a = Image.fromarray(h_result)
	a.save('out.jpeg')
	
	a.show()
	