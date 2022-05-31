import cv2
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt

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

def estraiSilhouetteVecchio(image):

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
    return res

def estraiSilhouette(image):

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
    #cv2.imshow("original", image)
    #cv2.imshow("res", res)
    return res

def thresh(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,th1 = cv2.threshold(image,90,255,cv2.THRESH_BINARY)
    ret, th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(image,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("original", image)
    cv2.imshow("res", th3)

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
    line = createLineIterator(np.array(pointA), np.array(pointB), image)
    center1 = int(0.19*len(line))
    center2 = int(0.34*len(line))
    center3 = int(0.5*len(line))
    center4 = int(0.67*len(line))
    center5 = int(0.83*len(line))
    num = coloreABinario(line,center1) + coloreABinario(line,center2) + coloreABinario(line,center3) + coloreABinario(line,center4) + coloreABinario(line,center5)   
    return int(num,2)

def nuoveCoordinateA(markNum):
    shift = -15
    radius = 70
    angle = np.radians(shift*markNum)
    nx = np.cos(angle)*radius 
    ny = np.sin(angle)*radius
    return nx, ny, 0    # z=0

def nuoveCoordinateB(markNum):
    shift = -15
    radius = np.sqrt(65*65 + 5*5)
    angle = np.radians(shift*markNum)
    nx = np.cos(angle)*radius 
    ny = np.sin(angle)*radius
    return nx, ny, 0    # z=0

def pulisciContorni(contorni):
    approxContorni = []
    area=0
    for cont in contorni:
        M = cv2.moments(cont)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center = np.array([cx,cy])
        approx_c = cv2.approxPolyDP(cont,5,True)
        #approx_c = cv2.approxPolyDP(cont, 0.01 * cv2.arcLength(cont, True), True)
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
    approxContorni = list(c for c in approxContorni if cv2.contourArea(c[0])>avg_area)      #era > avg_area/1.7 ma "sfarfallava" a volte
    marksCont = tuple(c[0] for c in approxContorni)
    indexConcave = tuple(c[1] for c in approxContorni)

    return indexConcave, marksCont

def estraiPunti(marksCont, indexConcave, image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image_bn = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    objPoints = np.zeros((len(marksCont),3), np.float32)
    imgPoints = np.zeros((len(marksCont),2), np.float32)
    for c,cont in enumerate(marksCont):    
        #traccio una linea
        x1, y1 = cont[(indexConcave[c]+2)%5][0]
        x2, y2 = cont[(indexConcave[c]+3)%5][0]    
        midX = int((x1 + x2) / 2)
        midY = int((y1 + y2) / 2)
        pointA = (midX, midY)
        cx, cy = cont[indexConcave[c]][0]
        pointB = (cx, cy)
        num = numeraMark(image_bn, pointA, pointB)
        x, y = cont[(indexConcave[c]+2)%5][0]
        cv2.putText(image, str(num), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        ox, oy, oz = nuoveCoordinateA(num)
        objPoints[c] = np.float32([ox,oy,oz])
        imgPoints[c] = np.float32([cx, cy])
    return objPoints,imgPoints

def calcolaRMS(predictions, targets):
    xErr = np.sqrt(np.mean((predictions[0]-targets[0])**2))
    yErr = np.sqrt(np.mean((predictions[1]-targets[1])**2))
    return xErr, yErr

def poseEstimation(image, mtx, dist):


    contorni = estraiContorni(image)
    indexConcave, marksCont = pulisciContorni(contorni)
    cv2.drawContours(image, contours=marksCont, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    objPoints,imgPoints = estraiPunti(marksCont, indexConcave, image)
    #print("Punti usati: " + str(len(objPoints)))
    ret, rvecs, tvecs = cv2.solvePnP(objPoints, imgPoints, mtx, dist, cv2.SOLVEPNP_IPPE)

    cs = 60
    up = 80
    axisBoxes = np.float32([[-cs,-cs,cs*2+up], [+cs,-cs,cs*2+up], [+cs,+cs,cs*2+up], [-cs,+cs,cs*2+up],
                                [-cs,-cs,up],    [+cs,-cs,up],    [+cs,+cs,up],    [-cs,+cs,up]])
    
    imgpts, jac = cv2.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)
    """ projImgPts, jac = cv2.projectPoints(objPoints, rvecs, tvecs, mtx, dist)
    xErr, yErr = calcolaRMS(projImgPts, imgPoints)
    if xErr>1 or yErr>1:
        print("RMS maggiore di 1! (" + str(xErr) + ", " + str(yErr) + ")") """
    #img = drawBoxes(image, imgPoints, imgpts)
    #cv2.imshow("borders", img)
    return rvecs, tvecs

def getVoxelsCenters(nVoxels, side, topLeft):
    centers = np.zeros((nVoxels**3,3), np.float32)
    shift = side*2/nVoxels
    for i in range(nVoxels):
        for j in range(nVoxels):
            for k in range(nVoxels):
                index = i*nVoxels**2 + j*nVoxels + k
                cx = topLeft[0] - i*shift - shift/2
                cy = topLeft[1] + k*shift + shift/2
                cz = topLeft[2] - j*shift - shift/2
                centers[index] = np.float32([cx,cy,cz])
    return centers
                

def carve(image, voxelCenters, voxels, mtx, dist, DRAW): 
    rvecs, tvecs = poseEstimation(image, mtx, dist)
    sil = estraiSilhouette(image)
    #sil = rimuoviBG(image, grayMedianFrame)
    #print(len(voxelCenters), flush=True)
    imgPts, jac = cv2.projectPoints(voxelCenters, rvecs, tvecs, mtx, dist)
    #removed = 0
    for p, point in enumerate(imgPts):
        x = int(point[0][0])
        y = int(point[0][1])
        if sil[y][x] == 0:    #nero
            #np.delete(voxelCenters, p-removed)
            #removed += 1
            voxels[p] = False
        if  DRAW and voxels[p] ==  True:
            cv2.drawMarker(image, (x, y),(0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1, line_type=cv2.LINE_AA)
    cv2.imshow("borders", image)
    #return voxelCenters
    return voxels

    
def saveToPLY(name, voxels, voxelCenters):
    nPoints = np.sum(voxels)
    fullName = "results" + name + ".ply"
    f = open(fullName, "w")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex " + str(nPoints) +"\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")
    for ind, vc in enumerate(voxelCenters):
        if voxels[ind] == True:
            f.write(str(vc[0]) + " " + str(vc[1]) + " " + str(vc[2]) + "\n")
    f.close()

nomeFile = "\obj01"
vidcap = cv2.VideoCapture("data" + nomeFile + ".mp4")
#ret, image = vidcap.read()

with open('mtx.pkl', 'rb') as f:
    mtx = pickle.load(f)
with open('dist.pkl', 'rb') as f:
    dist = pickle.load(f)

nVox = 50
side = 60
up = 80
voxels = np.full((nVox**3), True)
cubeTopLeftCorner = np.float32([+side,-side, side*2+up]) #coordinate angolo in alto a sinistra del cubo grande (facciata frontale)
voxelCenters = getVoxelsCenters(nVox, side, cubeTopLeftCorner)
#carve(image, nVox, voxelCenters, voxels, mtx, dist)
frameCounter = 0


vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:    
    ret, image = vidcap.read()
    frameCounter += 1
    if not ret:
        break

    if frameCounter%10 == 0:
        #voxelCenters = carve(image, voxelCenters, voxels, mtx, dist, DRAW=False)
        #voxels = carve(image, voxelCenters, voxels, mtx, dist, DRAW=True)
        thresh(image)
        #res, backSub = estraiSilhouette(image, backSub)
    k = cv2.waitKey(1) & 0xff    
    # Check if 'q' key is pressed.
    if k == ord('q'):
        break

# Release the VideoCapture Object.
print("Fine Carving")
vidcap.release()
saveToPLY(nomeFile, voxels, voxelCenters)
cv2.destroyAllWindows()
