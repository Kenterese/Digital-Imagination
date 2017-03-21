# NumPy is the fundamental package for scientific computing with Python.
import numpy as np
import cv2
# matplotlib is a python 2D plotting library 
import matplotlib.pyplot as plt

img1 = cv2.imread('2.png ',0)
img2 = cv2.imread('opencv-feature-matching-image.jpg',0)


# This is the detector we're going to use for the features.
orb = cv2.ORB()

#Here, we find the key points and their descriptors with the orb detector.
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

print "kp1", kp1
# array of keypoint consist of x n y	

print "kp2", kp2
# 

print "des1", des1
# 

print "des2", des2	


#This is our BFMatcher object.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)



# create matches of the descriptors, then we sort them based on their distances.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()

