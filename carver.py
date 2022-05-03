import cv2
import numpy as np
import glob

vidcap = cv2.VideoCapture('data\obj01.mp4')
vidcap.set(1, 50)
ret, image = vidcap.read()
image = image[:,1020:1800]
image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
# visualize the binary image
#cv2.imshow('Binary image', thresh)
#cv2.waitKey(0)
#cv2.imwrite('image_thres1.jpg', thresh)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                     
# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
               
# see the results
#cv2.imshow('None approximation', image_copy)
#cv2.waitKey(0)
#cv2.imwrite('contours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()

#pts1 = np.float32([[0, 400], [780, 400],
                       #[0, 1080], [780, 1080]])
pts1 = np.float32([[0, 10], [880, 120],
                       [0, 600], [880, 600]])
pts2 = np.float32([[0, 0], [400, 0],
                    [0, 640], [400, 640]])
    
# Apply Perspective Transform Algorithm
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(image, matrix, (500, 600))
    
# Wrap the transformed image
""" cv2.imshow('frame', image) # Initial Capture
cv2.waitKey(0)
cv2.imshow('frame1', result)
cv2.waitKey(0)
cv2.destroyAllWindows() """


while True:
    
    # Read a new frame.
    ret, image = vidcap.read()

    # Check if frame is not read correctly.
    if not ret:
        
        # Break the loop.

        break
    
    image = image[:,1020:1800]
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    
    pts1 = np.float32([[0, 120], [880, 120],
                       [0, 600], [880, 600]])
    pts2 = np.float32([[0, 0], [400, 0],
                        [0, 640], [400, 640]])
        
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (500, 600))
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = result.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # Display
    cv2.imshow("original", image)
    cv2.imshow("res", image_copy)

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
#cv2.imshow("image", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows() 
