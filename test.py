import cv2
import numpy as np
import matplotlib.pyplot as plt

##############################################

#################
# LOADING IMAGE #
#################

img = cv2.imread("./leonardo-dicaprio-young-1993.jpg", cv2.IMREAD_UNCHANGED)
#IMREAD_COLOR = 1
#IMREAD_GRAYSCALE = 0
#IMREAD_UNCHANGED = -1

#cv2.imshow("image", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#plt.imshow(img, cmap="gray", interpolation="bicubic")
#plt.plot([50, 100], [80, 100], 'c', linewidth=5)
#plt.show()

####################################################

################################
# DRAWING AND WRITING ON IMAGE #
################################

# Draw a line
            #Start      #End        #Color      #Linewidth
# cv2.line(img, (0, 0), (150, 150), (255, 255, 255), 15)

# # Draw a rectangle
# cv2.rectangle(img, (15, 25), (200, 150), (0, 255, 0), 5)

# # Draw a circle
#             # Center   #Radius   #Color     #Thickness
# cv2.circle(img, (100, 63), 55, (0, 0, 255), -1)

# # Draw polygons
# pts = np.array([[10, 5], [80, 300], [70, 20], [50, 10]], np.int32)
# cv2.polylines(img, [pts], True, (0, 255, 255), 3)


# # Add text
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, "OpenCV", (0, 130), font, 1, (200, 255, 255), 2, cv2.LINE_AA)

###################################

####################
# IMAGE OPERATIONS #
####################

# Change value of specific pixel
# img[55, 55] = [255, 255, 255]
# px = img[55, 55]
# print(px)

# # Change value of a region of an image
# img[100:150, 100:150] = [255, 255, 255]
# roi = img[100:150, 100:150]
# print(roi)

# # Copy and paste pixels
# watch_face = img[300:374, 307:394]
# img[0:74, 0:87] = watch_face

##############################

# IMAGE ARITHMETIC AND LOGIC

# img1 = cv2.imread("yeah3.jpg")
img2 = cv2.imread("wu3.jpg")

# # This is numpy addition
# # Add pixels to pixels element-wise with modulo operation
# # Ex: 255+ 10 = 260 % 256 => 4
# #add = img1 + img2

# # This is OpenCV addition
# # Add pixels to pixels element-wise with saturated operation
# # Ex: 250+10 = 260 => 255 
# #add = cv2.add(img1, img2)

# # Blending 2 images
#                     # Img1 #weight1 #Img2 #weight2 #bias
# #weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

# # Insert img2 into img1
# rows, cols, channels = img2.shape
# indent = 180
# roi = img1[-rows:, indent:indent+cols]
# # Convert color
# img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# # pixels > 220, will be converted to 0, otherwise 255
# # mask contains True in the foreground, False in the background
# ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY_INV)

# # mask_inv contains True in the background, False in the foreground
# mask_inv = cv2.bitwise_not(mask)

# #ret, mask = cv2.threshold(img2gray, 128, 255, cv2.THRESH_BINARY)

# # operate bitwise and when the corresponding pixel in the mask is True
# # keep the background of the img1 and add the black foreground of img2 to the img1
# img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# # keep the foreground of the img2
# img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

# # add them together to get the foreground of img2 and the background of img1
# dst = cv2.add(img1_bg, img2_fg)

# # assign it to the original img1
# img1[-rows:, indent:indent+cols] = dst


#########################################

################
# THRESHOLDING #
################

# img3 = cv2.imread("dark.jpg")

# retval, threshold = cv2.threshold(img3, 70, 255, cv2.THRESH_BINARY)

# grayscaled = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
# retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)

# # threshold value is calculated locally for each pixel, instead of using a global fixed value for the entire image.
# # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)

# gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
# # maxValue: The maximum value that will be assigned to pixels that pass the threshold.
# # adaptiveMethod: The method used to calculate the local threshold value. It can take two values:
# #   cv2.ADAPTIVE_THRESH_MEAN_C: The threshold value is the mean of the neighborhood area.
# #   cv2.ADAPTIVE_THRESH_GAUSSIAN_C: The threshold value is the weighted sum of the neighborhood values where weights are given by a Gaussian window.
# # blockSize: The size of the neighborhood area in pixels. It should be an odd integer
# # C: A constant subtracted from the mean or weighted sum to fine-tune the threshold value. (bias)

# retval3, otsu = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


####################################
#mask_skin = cv2.inRange()
dark_vest = (0, 0, 0)
light_vest = (130, 200, 165)

lab_img = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
mask_vest = cv2.inRange(lab_img, dark_vest, light_vest)

hsv_img = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
dark_skin = (5, 42, 180)
light_skin = (15, 123, 240)
mask_skin = cv2.inRange(hsv_img, dark_skin, light_skin)

cv2.imshow("mask", mask_skin)
cv2.imshow("vest", mask_vest)
cv2.imshow("image", hsv_img)
#cv2.imshow("mask", mask_inv)
# waitKey(0) will display the window infinitely until any keypress 
# (it is suitable for image display).
cv2.waitKey(0)
cv2.destroyAllWindows()