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
	
	_global_ void add_num(float *d_result, float *d_a, float *d_b, int N){
		int tid = threadIdx.x + blockDim.x*blockIdx.x;
		while (tid < N){
			d_result[tid] = d_a[tid] + d_b[tid];
			if (d_result[tid] > 255)
				d_result[tid] = 255;
			tid += (blockDim.x * gridDim.x);
		}
	
	}	
	""
	)
	
	img1 = cv2.imread('2.jpeg',0)
	img2 = cv2.imread('out1.jpeg',0)
	h_img1 = img1.reshape(60000).astype(np.float32)
	h_img2 = img2.reshape(60000).astype(np.float32)
	N = h_img1.size
	h_result = h_img1
	add_img = mod.get_function("add_num")
	add_img(drv.Out(h_result), drv.In(h_img1), drv.In(h_img2), block=(1024, 1, 1), grid=(64, 1, 1))
	
	h_result = np.reshape(h_result,(300,200)).astype(np.uint8)
	
	a = Image.fromarray(h_result)
	a.save('out3.jpeg')
	
	a.show()