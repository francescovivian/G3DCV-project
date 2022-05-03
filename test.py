import cv2
import numpy as np
import glob

vidcap = cv2.VideoCapture('data\obj01.mp4')
#vidcap.set(1, 50)
#ret, image = vidcap.read()
#image = image[240:850,420:1000]

#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Perform color-segmentation to get the binary mask
#lwr = np.array([0, 0, 0])
#upr = np.array([179, 255, 146])
#msk = cv2.inRange(hsv, lwr, upr)

# Extracting the rod using binary-mask
#krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
#dlt = cv2.dilate(msk, krn, iterations = 1)
#dlt = cv2.erode(dlt, krn, iterations = 10)

#res = 255 - cv2.bitwise_and(dlt, msk)

# Display
#cv2.imshow("original", image)
#cv2.waitKey(0)
#cv2.imshow("res", res)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

while True:
    
    # Read a new frame.
    ret, image = vidcap.read()

    # Check if frame is not read correctly.
    if not ret:
        
        # Break the loop.

        break
    
    image = image[240:850,420:1000]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Perform color-segmentation to get the binary mask
    lwr = np.array([0, 0, 0])
    upr = np.array([179, 255, 146])
    msk = cv2.inRange(hsv, lwr, upr)

    # Extracting the rod using binary-mask
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dlt = cv2.dilate(msk, krn, iterations = 1)
    dlt = cv2.erode(dlt, krn, iterations = 10)

    res = 255 - cv2.bitwise_and(dlt, msk)

    # Display
    cv2.imshow("original", image)
    cv2.imshow("res", res)

    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xff
    
    # Check if 'q' key is pressed.
    if k == ord('q'):
        
        # Break the loop.
        break

# Release the VideoCapture Object.
vidcap.release()

# Close the windows.q
cv2.destroyAllWindows()