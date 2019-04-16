import pycuda.driver as drv
import numpy as np
import cv2
import pycuda.gpuarray as gpuarray
from PIL import Image
import pycuda.autoinit

img = cv2.imread('out.jpeg',0)
h_img = img.reshape(60000).astype(np.float32)
d_img = gpuarray.to_gpu(h_img)
d_result = 255 - d_img
h_result = d_result.get()
h_result = np.reshape(h_result,(300,200)).astype(np.unint8)

a = Image.fromarray(h_result)
a.save('out1.jpeg')

a.show()