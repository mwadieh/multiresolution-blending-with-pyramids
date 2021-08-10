# -*- coding: utf-8 -*-
"""
Created on Sun May  2 03:13:39 2021

@author: mwadieh
"""


import cv2
import numpy as np,sys

a = cv2.imread('apple.JPEG')
o = cv2.imread('orange.JPEG')

# downsampling of image
def downsample(image):
    i,j,k = image.shape
    image2 = image.copy()
    delete_rows = []
    delete_cols = []
    
    for ii in range(i):
        if ii%2==1:
          delete_rows.append(ii-1)
    
    for jj in range(j):
        if jj%2==1:
          delete_cols.append(jj-1)
    
    image2 = cv2.GaussianBlur(image,(3,3),0)
    image_residual = image2-image # for that stage
    image2 = np.delete(image2,delete_rows,axis=0)
    image2 = np.delete(image2,delete_cols,axis=1)
    
    
    
    return image2,image_residual

# create gaussian pyramid
# a1 = cv2.GaussianBlur(a,(5,5),0)

    
def gaussian_pyramid(image):
    
    img0=image
    img1 = downsample(image)[0]
    img2 = downsample(img1)[0]
    img3 = downsample(img2)[0]
    img4 = downsample(img3)[0]
    
    return img0,img1,img2,img3,img4


def laplacian_pyramid(image):
   
    img0,img1,img2,img3,img4 = gaussian_pyramid(image)
    
    lap0 = downsample(img0)[1]
    lap1 = downsample(img1)[1]
    lap2 = downsample(img2)[1]
    lap3 = downsample(img3)[1]
    lap4 = img4
    
    return lap0,lap1,lap2,lap3,lap4 


def recover(image):
    
    img0,img1,img2,img3,img4 = gaussian_pyramid(image)
    lap0,lap1,lap2,lap3,lap4 = laplacian_pyramid(image)
    
    # reconstructing 3 
    img3_new = cv2.resize(img4,(img3.shape[0],img3.shape[1]))
    img3_new = cv2.GaussianBlur(img3_new,(3,3),0)+lap3
    
    # reconstructing 2
    img2_new = cv2.resize(img3_new,(img2.shape[0],img2.shape[1]))
    img2_new = cv2.GaussianBlur(img2_new,(3,3),0)+lap2
    
     # reconstructing 1
    img1_new = cv2.resize(img2_new,(img1.shape[0],img1.shape[1]))
    img1_new = cv2.GaussianBlur(img1_new,(3,3),0)+lap1
    
     # reconstructing 0
    img0_new = cv2.resize(img1_new,(img0.shape[0],img0.shape[1]))
    img0_new = cv2.GaussianBlur(img0_new,(3,3),0)+lap0
    
    return img0_new


# creating a mask
mask = np.zeros(o.shape[:2], dtype="uint8")
mask = np.array([mask,mask,mask])
plt.title('Direct Blending')
mask = mask.reshape(o.shape)

# I = (1 - M) * I1 + M * I2,

# direct blending
direct_blend = np.hstack((o[:,:int(o.shape[1]/2)],a[:,int(o.shape[1]/2):]))
qq = cv2.cvtColor(direct_blend, cv2.COLOR_BGR2RGB)
plt.imshow(qq)
plt.show()


# alpha blending
o_mask = np.hstack((o[:,:int(o.shape[1]/2)],mask[:,int(o.shape[1]/2):]))
o_mask = cv2.GaussianBlur(o_mask,(5,5),0)
plt.imshow(o_mask)
plt.title('Alpha orange_mask')
plt.show()

a_mask = np.hstack((mask[:,:int(o.shape[1]/2)],a[:,int(o.shape[1]/2):]))
a_mask = cv2.GaussianBlur(a_mask,(5,5),0)
plt.imshow(a_mask)
plt.title('Alpha apple_mask')
plt.show()

alpha_blend = np.hstack((o_mask[:,:int(o.shape[1]/2)],a_mask[:,int(o.shape[1]/2):]))
plt.imshow(alpha_blend)
plt.title('Alpha Blending 1')
plt.show()


alpha_blend = cv2.GaussianBlur(direct_blend,(5,5),0)
qq = cv2.cvtColor(alpha_blend, cv2.COLOR_BGR2RGB)
plt.imshow(qq)
plt.title('Alpha Blending 2')
plt.show()

# function for joining
def join(image1,image2):
    join_image = np.hstack((image1[:,:int(image1.shape[1]/2)],image2[:,int(image1.shape[1]/2):]))
    return join_image


# for reconstructing image
def reconstruct(previous_gaussian,current_gaussian,laplacian):
    # reconstructing 3 
    img_reconstructed = cv2.resize(previous_gaussian,(current_gaussian.shape[0],current_gaussian.shape[1]))
    img_reconstructed = cv2.GaussianBlur(img_reconstructed,(3,3),0)+laplacian
    
    return img_reconstructed


def multiblend(image1,image2):
    
 #   guassian pyramid for image 1 and 2
    img1_0,img1_1,img1_2,img1_3,img1_4 = gaussian_pyramid(image1)
    img2_0,img2_1,img2_2,img2_3,img2_4 = gaussian_pyramid(image2)
     
 #  laplacian pyramid for image 1 to 4
    lap1_0,lap1_1,lap1_2,lap1_3,lap1_4 = laplacian_pyramid(image1)
    lap2_0,lap2_1,lap2_2,lap2_3,lap2_4 = laplacian_pyramid(image2)
 #   joining gaussians
    img0 = join(img1_0,img2_0)
    img1 = join(img1_1,img2_1)
    img2 = join(img1_2,img2_2)
    img3 = join(img1_3,img2_3)
    img4 = join(img1_4,img2_4)
 #  joining laplacians    
    lap0 = join(lap1_0,lap2_0)
    lap1 = join(lap1_1,lap2_1)
    lap2 = join(lap1_2,lap2_2)
    lap3 = join(lap1_3,lap2_3)
    lap4 = join(lap1_4,lap2_4)
    # reconstructing 3 
    img3_new = reconstruct(img4,img3,lap3)
    # reconstructing 2
    img2_new = reconstruct(img3_new,img2,lap2)
     # reconstructing 1
    img1_new = reconstruct(img2_new,img1,lap1)
     # reconstructing 0
    img0_new = reconstruct(img1_new,img0,lap0)
     
    return img0_new

qq = multiblend(o,a)
plt.imshow(qq)
plt.show()

qq = cv2.cvtColor(qq, cv2.COLOR_BGR2RGB)
plt.imshow(qq)
plt.title('Multi Resolution Blending')
plt.show()

kente = cv2.imread('kente.JPG')
serape = cv2.imread('serape_mexican.JPG')

kente = kente[0:400,0:400]
serape = serape[0:400,0:400]
plt.imshow(cv2.cvtColor(kente, cv2.COLOR_BGR2RGB))
plt.title('Kente')
plt.show()

plt.imshow(cv2.cvtColor(serape, cv2.COLOR_BGR2RGB))
plt.title('Serape')
plt.show()



image_blend = multiblend(kente,serape)
image_blend = cv2.cvtColor(image_blend, cv2.COLOR_BGR2RGB)
plt.imshow(image_blend)
plt.title('Blended Kente and Serape Image')
plt.show()
