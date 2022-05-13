import cv2
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 10)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 10)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 10)
    return img

def drawBoxes(img, corners, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img


def estimatePose(image, mtx, dist):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((24*17,3), np.float32)
    objp[:,:2] = np.mgrid[0:24,0:17].T.reshape(-1,2)
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    axisBoxes = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                    [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    pts = np.float32([[0,0], [0,50],
                      [50,0], [50,50]])
    #ret, corners = cv2.findChessboardCorners(gray, (24,17),None)
    
    #corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    ret, rvecs, tvecs = cv2.solvePnP(objp, pts, mtx, dist)
    imgpts, jac = cv2.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)

    img = drawBoxes(image, pts, imgpts)
    cv2.imshow('img',img)
    return

def estraiSilhouette(image):
    image = image[240:850,420:930]
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)

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

def disegnaBordiDisco(image):
    image = image[:,1020:1800]
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    disegnaBordi(image)

def trasformaProspettiva(image):
    pts1 = np.float32([[0, 125], [880, 125],
                       [0, 420], [880, 420]])
    pts2 = np.float32([[0, 0], [400, 0],
                        [0, 400], [400, 400]])
        
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (500, 600))
    return result

def disegnaBordi(image):
    result = trasformaProspettiva(image)
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = result.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # Display
    cv2.imshow("original", image)
    cv2.imshow("borders", image_copy)


def carving(image):
    image = image[140:950,320:1030]
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    start_point = (100, 50)
    end_point = (730, 680)
    ticks = 2
    w = end_point[0] - start_point[0]
    #h = end_point[1] - start_point[1]
    step = round(w/ticks)
    color = (0, 0, 255)
    thickness = 1
    for i in range(ticks):
        for j in range(ticks):
            start = (start_point[0]+i*step, start_point[1]+j*step)
            end = (start_point[0]+(i+1)*step, start_point[1]+(j+1)*step)
            image = cv2.rectangle(image, start, end, color, thickness)
    cv2.imshow('box', image) 

def estraiContorni(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return contours

def getAngle(a, b, c):
    ba = a - b
    bc = c - b
    #print(ba)
    cosine_angle = np.dot(ba, bc.transpose()) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.round(np.degrees(angle))

def poseEstimation(image):
    image_copy = image.copy()
    contorni = estraiContorni(image)
    approxContorni = []
    area=0
    for cont in contorni:
        M = cv2.moments(cont)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center = np.array([cx,cy])
        approx_c = cv2.approxPolyDP(cont,5,True)
        if len(approx_c) == 5:
            convex = concave = 0
            for i in range(5):
                #x, y = approx_c[i][0]
                angle = getAngle(center, np.array(approx_c[i%5]), np.array(approx_c[(i+1)%5])) + getAngle(np.array(approx_c[(i-1)%5]), np.array(approx_c[i%5]), center)
                #cv2.putText(image_copy, str(angle), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)
                if angle < 180:
                    convex += 1
                elif angle >180:
                    indexC = i
                    concave +=1
            if convex == 4 and concave==1:
                area += cv2.contourArea(approx_c)
                approxContorni.append([approx_c, indexC])

    avg_area = area/len(approxContorni)
    approxContorni = list(c for c in approxContorni if cv2.contourArea(c[0])>avg_area/3)
    marksCont = tuple(c[0] for c in approxContorni)
    indexConcave = tuple(c[1] for c in approxContorni)

    cv2.drawContours(image=image_copy, contours=marksCont, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    for c,cont in enumerate(marksCont):
        for i in range(5):
            x, y = cont[(i+indexConcave[c])%5][0]
            cv2.putText(image_copy, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    
    # Display
    cv2.imshow("borders", image_copy)

def matchMarker(image, marker):

    image = image[:,1020:1800]
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)

    result = trasformaProspettiva(image)
    contorniDisco = estraiContorni(result)
    contornoMarker = estraiContorni(marker)[1]
    contorniOk = ()
    for cont in contorniDisco:
        ret = cv2.matchShapes(cont,contornoMarker,1,0.0)
        if ret < 0.2:
            contorniOk = contorniOk + (cont,)
    #cv2.imshow('result', result)
    cv2.drawContours(image=result, contours=contorniOk, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # Display
    cv2.imshow("original", image)
    cv2.imshow("borders", result)

def featureMatcher(image, mark):
    image = image[:,1020:1800]
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)

    image = trasformaProspettiva(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mark = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)
    #ret, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image,None)
    kp2, des2 = sift.detectAndCompute(mark,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(image,kp1,mark,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("result",img3)
    #,plt.show()

vidcap = cv2.VideoCapture('data\obj01.mp4')
#ret, image = vidcap.read()
marker = cv2.imread("mark3.png")
#featureMatcher(image,marker)
#poseEstimation(image)

with open('mtx.pkl', 'rb') as f:
    mtx = pickle.load(f)
with open('dist.pkl', 'rb') as f:
    dist = pickle.load(f)
#estraiContorni(marker)
#matchMarker(image, marker)
while True:    
    ret, image = vidcap.read()
    if not ret:
        break
    #estraiSilhouette(image)
    #disegnaBordiDisco(image)
    #estimatePose(image,mtx,dist)
    #carving(image)
    #matchMarker(image, marker)
    #featureMatcher(image,marker)
    poseEstimation(image)
    k = cv2.waitKey(1) & 0xff    
    # Check if 'q' key is pressed.
    if k == ord('q'):
        break

# Release the VideoCapture Object.
vidcap.release()

cv2.destroyAllWindows()
