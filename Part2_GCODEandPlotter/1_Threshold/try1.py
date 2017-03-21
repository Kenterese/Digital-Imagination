import cv2
import numpy as np

img = cv2.imread ('/Users/Kevinpro/Desktop/PythonOpenCV/EdgeDetection/30.jpg', cv2.IMREAD_GRAYSCALE)
#edges = cv2.Canny(img,200,200)

th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,15,2)

#cv2.imshow('image', img)
#cv2.imshow('image2', edges)
cv2.imshow('image3',th3)
#cv2.imwrite('output.jpg',th3)

cv2.waitKey(0)
cv2.destoryAllWindows()