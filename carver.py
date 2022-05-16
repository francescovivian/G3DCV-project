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

def createLineIterator(P1, P2, img):
    """
    Iterator implementation found on the web
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer


def getAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc.transpose()) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.round(np.degrees(angle))


def coloreABinario(line, pos):
    if(line[pos][2]==255):
        return "0"
    else:
        return "1"

def numeraMark(image, pointA, pointB):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, img_gray = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    #cv2.imshow("bn", img_gray)
    line = createLineIterator(np.array(pointA), np.array(pointB), img_gray)
    center1 = int(0.19*len(line))
    center2 = int(0.34*len(line))
    center3 = int(0.5*len(line))
    center4 = int(0.67*len(line))
    center5 = int(0.83*len(line))
    num = coloreABinario(line,center1) + coloreABinario(line,center2) + coloreABinario(line,center3) + coloreABinario(line,center4) + coloreABinario(line,center5)   
    return int(num,2)

def nuoveCoordinateA(markNum):
    shift = 15
    radius = 70
    angle = np.radians(shift*markNum)
    nx = np.cos(angle)*radius 
    ny = np.sin(angle)*radius
    return nx, ny, 0    # z=0

def nuoveCoordinateB(markNum):
    shift = 15
    radius = np.sqrt(65*65 + 5*5)
    angle = np.radians(shift*markNum)
    nx = np.cos(angle)*radius 
    ny = np.sin(angle)*radius
    return nx, ny, 0    # z=0

def poseEstimation(image):

    with open('mtx.pkl', 'rb') as f:
        mtx = pickle.load(f)
    with open('dist.pkl', 'rb') as f:
        dist = pickle.load(f)

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
                angle = getAngle(center, np.array(approx_c[i%5]), np.array(approx_c[(i+1)%5])) + getAngle(np.array(approx_c[(i-1)%5]), np.array(approx_c[i%5]), center)
                if angle < 180:
                    convex += 1
                else:
                    indexC = i
                    concave +=1
            if convex == 4 and concave==1:
                area += cv2.contourArea(approx_c)
                approxContorni.append([approx_c, indexC])

    avg_area = area/len(approxContorni)
    approxContorni = list(c for c in approxContorni if cv2.contourArea(c[0])>avg_area/1.7)
    marksCont = tuple(c[0] for c in approxContorni)
    indexConcave = tuple(c[1] for c in approxContorni)

    cv2.drawContours(image=image_copy, contours=marksCont, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    objPoints = np.zeros((len(marksCont),3), np.float32)
    imgPoints = np.empty((len(marksCont),2), np.float32)

    for c,cont in enumerate(marksCont):
        
        #traccio una linea
        x1, y1 = cont[(indexConcave[c]+2)%5][0]
        x2, y2 = cont[(indexConcave[c]+3)%5][0]    
        midX = int((x1 + x2) / 2)
        midY = int((y1 + y2) / 2)
        pointA = (midX, midY)
        cx, cy = cont[indexConcave[c]][0]
        pointB = (cx, cy)
        num = numeraMark(image_copy, pointA, pointB)
        x, y = cont[(indexConcave[c]+2)%5][0]
        cv2.putText(image_copy, str(num), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        ox, oy, oz = nuoveCoordinateA(num)
        objPoints[c] = np.float32([ox,oy,oz])
        imgPoints[c] = np.float32([cx, cy])
        #image_copy = cv2.line(image_copy, pointA, pointB, color=(255, 255, 0), thickness=2)
        
        #enumero i vertici
        """ for i in range(5):
            x, y = cont[(i+indexConcave[c])%5][0]
            cv2.putText(image_copy, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
     """

    # Display
    #print(objPoints)
    #print(imgPoints)
    ret, rvecs, tvecs = cv2.solvePnP(objPoints, imgPoints, mtx, dist, cv2.SOLVEPNP_IPPE)
    axisBoxes = np.float32([[0,0,0], [0,30,0], [30,30,0], [30,0,0],
                    [0,0,-30],[0,30,-30],[30,30,-30],[30,0,-30] ])
    
    imgpts, jac = cv2.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)
    img = drawBoxes(image, imgPoints, imgpts)
    #print(rvecs)
    #print(tvecs)
    cv2.imshow("borders", img)


vidcap = cv2.VideoCapture('data\obj01.mp4')
#ret, image = vidcap.read()
#test = cv2.imread("test.png") #mark con numero 9
#poseEstimation(image)


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
