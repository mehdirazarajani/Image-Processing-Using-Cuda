from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

image = misc.imread('a.jpg')

print(type(image))

width_col,height_col,dim_col = image.shape

image_gray  = misc.imread('a.jpg')
image_gray = np.dot(image_gray[...,:3],[0.299, 0.587, 0.114])
width_gray,height_gray = image_gray.shape

# image_gray_dir  = misc.imread('a.jpg',mode="L")

f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(image)
axarr[0, 0].set_title('Image Color')

axarr[0, 1].imshow(image_gray,cmap = plt.get_cmap('gray'))
axarr[0, 1].set_title('Image converted to Gray')

# image is read in 2 dimension only ---> black an white image 
# axarr[1, 0].imshow(image_gray_dir,cmap = plt.get_cmap('gray'))
# axarr[1, 0].set_title('Image directly read to gray')

plt.show() 

misc.imsave("black_and_white_image.jpg",image_gray)