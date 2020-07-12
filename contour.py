import numpy as np
import cv2
import matplotlib.pyplot as plt
# matplotlib qt

img = cv2.imread("hand.jpg")
img_copy= np.copy(img)
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
#plt.imshow(img_copy)
gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
#plt.imshow(gray)
edged = cv2.Canny(gray, 30, 200)

#Create binary threshold img
retval, binary = cv2.threshold( gray, 139, 255, cv2.THRESH_BINARY_INV)
plt.imshow(binary)

#Find Contours through threshold
contours, hierarchy = cv2.findContours( edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#we have out countours
#show them on img
img_copy2 = np.copy(img_copy)
countour_img = cv2.drawContours(img_copy2, contours, -1, (10, 215, 2), 40)
plt.imshow(countour_img)