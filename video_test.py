import cv2

import numpy as np

###########################################

#########################
# LOADING OFFLINE VIDEO #
#########################

# #cap = cv2.VideoCapture("./archive/videoplayback.mp4")
# # fourcc = cv2.VideoWriter_fourcc(*'XVID')
# # out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))
# while True:
#     ret, frame = cap.read()
#     print(ret)
#     cv2.imshow("frame", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break 

# cap.release()
# cv2.destroyAllWindows()

#######################################

##################
# LOADING WEBCAM #
##################

# #0 for first webcam
# #1 for second webcam

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #MJPG is Motion-JPEG

# while True:
#     status, frame = cap.read()
#     if status:
#         cv2.imshow("frame", frame)
#         # waitKey(1) will display a frame for 1 ms, 
#         # after which display will be automatically closed.
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         print(status)
#         break

##################################

###################
# COLOR FILTERING #
################### 

# cap = cv2.VideoCapture(-1)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# # HSV: Hue, Staturation Value
# # Hue is the root color (red, green, blue, yellow, purple, orange), 
# # Value is how dark or light the color is
# # Saturation is how gray a color is. 


# while True:
#     ret, frame = cap.read()
#     if ret:
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         cv2.imshow("frame", hsv)
        
#         lower_red = np.array([0, 20, 0])
#         upper_red = np.array([20, 180, 180])
        
#         mask = cv2.inRange(hsv, lower_red, upper_red)
#         res = cv2.bitwise_and(frame, frame, mask=mask)
#         cv2.imshow("mask", mask)
#         cv2.imshow("res", res)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        
#     else:
#         print(ret)
#         break    
    
##############################

##########################
# BLURRING AND SMOOTHING #
##########################

# cap = cv2.VideoCapture(-1)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


# while True:
#     ret, frame = cap.read()
#     if ret:
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         cv2.imshow("frame", hsv)
        
#         lower_red = np.array([0, 20, 0])
#         upper_red = np.array([20, 180, 180])
        
#         mask = cv2.inRange(hsv, lower_red, upper_red)
#         res = cv2.bitwise_and(frame, frame, mask=mask)
        
#         kernel = np.ones((15, 15), np.float32) / 225
        
#         # filter2D(src, ddepth, kernel)
#         # src: input image
#         # ddepth: desired depth (data type) of dest image. Use -1 to inidicate the depth should be the same as src image.
#         # kernel: convolutional kernel defining ther coefficients that will be multiplied with the image pixels during convolution.
#         smoothed = cv2.filter2D(res, -1, kernel)
        
#         # cv2.GaussianBlur(src, size, sigma)
#         # src: input image
#         # size: kernel size used to compute Gaussian weighted average for each pixel in the image.
#         # sigma: standard deviation of both X and Y directions.
#         blur = cv2.GaussianBlur(res, (15, 15), 0)
        
#         # cv2.medianBlur(src, ksize)
#         # src: input image
#         # ksize: the size of the median filter kernel. (should be odd number)
#         median = cv2.medianBlur(res, 15)
        
#         # Bilateral filter smoothening images and reducing noise, while preserving edges
#         # cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
#         # src: input image
#         # d: bilateral filter window size.
#         # sigmaColor: standard deviation of color space, the higher will result in more smoothing, the lower will preserve edges better.
#         # sigmaSpace: standard deviation of spatial space, the lower will more localized smoothing, the higher will reach farther pixels.
#         bilateral = cv2.bilateralFilter(res, 15, 75, 75)
        
#         cv2.imshow("mask", mask)
#         cv2.imshow("res", res)
        
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        
#     else:
#         print(ret)
#         break

##################################################

################################
# Morphological transformation #
################################

# cap = cv2.VideoCapture(-1)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


# while True:
#     ret, frame = cap.read()
#     if ret:
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         cv2.imshow("frame", hsv)
        
#         lower_red = np.array([0, 20, 0])
#         upper_red = np.array([20, 200, 200])
        
#         mask = cv2.inRange(hsv, lower_red, upper_red)
#         res = cv2.bitwise_and(frame, frame, mask=mask)
        
#         kernel = np.ones((5, 5), np.uint8)
#         erosion = cv2.erode(mask, kernel, iterations=1)
#         dilation = cv2.dilate(mask, kernel, iterations=1)
        
#         # opening is a combination of erosion followed by dilation, aim to remove false positives (noise in the background)
#         # closing is a combination of dilation followed by erosion, aim to remove false negatives (missing pixel in the foreground)

#         opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        
#         cv2.imshow("mask", mask)
#         cv2.imshow("res", res)
#         # cv2.imshow("erosion", erosion)
#         # cv2.imshow("dilation", dilation)
#         cv2.imshow("opening", opening)
#         cv2.imshow("closing", closing)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        
#     else:
#         print(ret)
#         break

########################################

################################
# Edge detection and gradients #
################################

# cap = cv2.VideoCapture(-1)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


# while True:
#     ret, frame = cap.read()
#     if ret:
#         # laplacian will convolve the src image with the sobel filter 
#         # to calculate the intensity around each pixels in order to detect edges:
#         # [
#         #     0  1  0
#         #     1 −4  1
#         #     0  1  0
#         # ]
        
#         # cv2.Laplacian(src, ddepth)
#         laplacian = cv2.Laplacian(frame, cv2.CV_64F)
#         sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
#         sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
        
#         # cv2.Canny(image, threshold1, threshold2)
#         # image: input image
#         # threshold1: The lower threshold value used in the hysteresis procedure.
#         # threshold2: The higher threshold value used in the hysteresis procedure.
#         edges = cv2.Canny(frame, 50, 100)
        
#         cv2.imshow("original", frame)
#         #cv2.imshow("sobelx", sobelx)
#         #cv2.imshow("sobely", sobely)
#         #cv2.imshow("laplacian", laplacian)
#         cv2.imshow("edge", edges)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        
#     else:
#         print(ret)
#         break

###########################################




cap.release()
cv2.destroyAllWindows()
