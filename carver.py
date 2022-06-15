from operator import truediv
import cv2
import numpy as np
import pickle
import os


""" 
Funzione che estrae la silhouette dell'oggetto da un frame.
    partendo dal seed (angolo 0,0 dell'immagine, parte dello sfondo), ogni pixel simile e "confinante" viene colorato
    del colore desiderato, in questo caso nero (0,0,0).

    nel caso del triceratopo, parte dello sfondo è tra le gambe e non confinante con il seed, seed2 è un punto tra le gambe
    da cui ripetere l'operazione. Nonostante il controllo sull'intensità, per oggetti che in quella posizione hanno un colore
    simile allo sfondo, cancella parte dell'oggetto.

    seed3 è un punto nel bicchiere, per poterlo "rimuovere" dall'immagine ma nel caso di oggetti con colore simile al bicchiere
    come ad esempio il pappagallo, estende il flood anche a parte dell'oggetto. 
"""
def estraiSilhouette(image):
    flood = image.copy()
    gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seed = (0, 0)       #sfondo
    seed2 = (840, 540)  #gambe triceratopo
    seed3 = (960, 540)  #bicchiere
    cv2.floodFill(flood, None, seedPoint=seed, newVal=(0, 0, 0), loDiff=(4, 4, 4, 4), upDiff=(4, 4, 4, 4))
    if (gs[540][840] < 75 and gs[540][840] > 65):
        cv2.floodFill(flood, None, seedPoint=seed2, newVal=(0, 0, 0), loDiff=(4, 4, 4, 4), upDiff=(4, 4, 4, 4))
    #cv2.floodFill(flood, None, seedPoint=seed3, newVal=(0, 0, 0), loDiff=(4, 4, 4, 4), upDiff=(4, 4, 4, 4))

    res = cv2.cvtColor(flood, cv2.COLOR_BGR2GRAY)

    # Display
    #cv2.imshow("test", res)
    return res

""" 
Funzione che dato un frame ritorna i contorni rilevati
"""
def estraiContorni(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return contours


"""
(implementazione da https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator, visto che non è presente in opencv2)

Funzione che dati due punti nell'immagine e l'immagine stessa, restituisce un array che rappresenta i punti nella linea tra P1 e P2
e per ogni punto (pixel) è registrata anche l'intensità.

    P1 è un numpy array contenente le coordinate del primo punto (x,y)
    P2 è un numpy array contenente le coordinate del secondo punto (x,y)
    img è l'immagine (in bianco e nero)
"""
def createLineIterator(P1, P2, img):
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

"""
Funzione che dati tre punti a, b, c, restituisce la misura dell'angolo con vertice in b.
    La misura dell'angolo ritornata è in gradi
"""
def getAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc.transpose()) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.round(np.degrees(angle))

"""
Funzione che data una posizione nella linea, restituisce 0 se il colore in quella posizione è bianco, 1 se è nero
"""
def coloreABinario(line, pos):
    if(line[pos][2]==255):
        return "0"
    else:
        return "1"


"""
Funzione che dati due punti di un mark (punto A e punto medio tra C e D), restituisce il numero di quel mark.
    La funzione controlla posizioni specifiche nella linea (centri dei "cerchi") per vedere se il colore è bianco o nero.
    Le percentuali della linea sono state ricavate analizzando il file marker.svg
"""
def numeraMark(image, pointA, pointB):
    line = createLineIterator(np.array(pointA), np.array(pointB), image)
    center1 = int(0.19*len(line))
    center2 = int(0.34*len(line))
    center3 = int(0.5*len(line))
    center4 = int(0.67*len(line))
    center5 = int(0.83*len(line))
    num = coloreABinario(line,center1) + coloreABinario(line,center2) + coloreABinario(line,center3) + coloreABinario(line,center4) + coloreABinario(line,center5)   
    return int(num,2)

"""
Funzione che dato un mark specifico, calcola le coordinate 3d del punto A di tale mark.
    Lo shift tra ogni mark è di -15° al crescere di ogni mark, per trovare lo shift totale tra il mark corrente ed il mark numero 0
    basta moltiplicare il numero del mark per -15°.

    la coordinata x sarà il coseno dell'angolo
    la coordinata y sarà il seno dell'angolo
    la coordinata z sarà fissa a 0, essendo il marker un piano
"""
def nuoveCoordinateA(markNum):
    shift = -15
    radius = 70
    angle = np.radians(shift*markNum)
    nx = np.cos(angle)*radius 
    ny = np.sin(angle)*radius
    return nx, ny, 0    # z=0

"""
Funzione che dati i contorni presenti in un' immagine, ritorna solamente quelli relativi ai marker
    Un contorno è mantenuto solamente se:
        -ha 5 angoli
        -ha 4 angoli convessi e 1 angolo concavo
        -ha un'area maggiore dell'area media di tutti gli altri contorni con 4 angoli convessi e 1 concavo (per evitare rumore)
    
    Per calcolare la misura di un angolo viene sfruttato il centro del contorno: per misurare l'angolo in A, si fa la somma
    dell'angolo E-A-Cent e dell'angolo B-A-Cent. Misurando direttamente l'angolo B-A-E si ottiene la misura dell'angolo esplementare
    (nel caso specifico dell'angolo A) e non si può distinguere se tale misura per un angolo generico è corretta oppure no.
"""
def pulisciContorni(contorni):
    approxContorni = []
    area=0
    for cont in contorni:
        M = cv2.moments(cont)                               #recupero il centro del contorno
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center = np.array([cx,cy])
        approx_c = cv2.approxPolyDP(cont,5,True)
        if len(approx_c) == 5:                              #considero solo contorni con 5 angoli
            convex = concave = 0
            for i in range(5):                              #per ogni angolo
                angle = getAngle(center, np.array(approx_c[i%5]), np.array(approx_c[(i+1)%5])) + getAngle(np.array(approx_c[(i-1)%5]), np.array(approx_c[i%5]), center)
                if angle < 180:
                    convex += 1
                else:                                       #questo è l'unico angolo concavo del mark
                    indexC = i                              #salvo l'indice dell'angolo concavo
                    concave +=1
            if convex == 4 and concave==1:                  #se ho 4 convessi e 1 concavo
                area += cv2.contourArea(approx_c)           #aggiungo all'area totale dei contorni con queste caratteristiche
                approxContorni.append([approx_c, indexC])   #lo salvo come contorno ok, assieme all'indice

    avg_area = area/len(approxContorni)                     #area media dei contorni salvati
    approxContorni = list(c for c in approxContorni if cv2.contourArea(c[0])>avg_area)      #area troppo piccola = rumore
    marksCont = tuple(c[0] for c in approxContorni)         #contorni dei mark finali
    indexConcave = tuple(c[1] for c in approxContorni)      #indice dell'angolo concavo di ogni mark

    return indexConcave, marksCont


"""
Funzione che dati i contorni dei marker, crea i vettori di image points e object points per poter fare poi la pose estimation.
    per ogni mark vengono calcolate le coordinate (immagine e oggetto) del punto A, cioè l'angolo concavo,
    per essere inserite nei vettori.
"""
def estraiPunti(marksCont, indexConcave, image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image_bn = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    objPoints = np.zeros((len(marksCont),3), np.float32)
    imgPoints = np.zeros((len(marksCont),2), np.float32)
    for c,cont in enumerate(marksCont):    
        x1, y1 = cont[(indexConcave[c]+2)%5][0]     #punto C
        x2, y2 = cont[(indexConcave[c]+3)%5][0]     #punto D
        midX = int((x1 + x2) / 2)                   #punto medio tra C e D
        midY = int((y1 + y2) / 2)
        pointA = (midX, midY)                       #primo estremo della linea
        cx, cy = cont[indexConcave[c]][0]           #angolo concavo (A)
        pointB = (cx, cy)                           #secondo estremo della linea
        num = numeraMark(image_bn, pointA, pointB)  #ricavo il numero del mark
        x, y = cont[(indexConcave[c]+2)%5][0]
        cv2.putText(image, str(num), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        ox, oy, oz = nuoveCoordinateA(num)          #coordinate dell punto A del mark corrente
        objPoints[c] = np.float32([ox,oy,oz])
        imgPoints[c] = np.float32([cx, cy])
    return objPoints,imgPoints


"""
Funzione che calcola l'errore RMS tra la proiezione dei punti e i punti attesi.
"""
def calcolaRMS(predictions, targets):
    xErr = np.sqrt(np.mean((predictions[0]-targets[0])**2))
    yErr = np.sqrt(np.mean((predictions[1]-targets[1])**2))
    return xErr, yErr

"""
Funzione che data un'immagine e i parametri "mtx" e "dist" ottenuti dalla calibrazione della camera,
stima la posizione corrente della camera rispetto all'oggetto.
"""
def poseEstimation(image, mtx, dist):
    contorni = estraiContorni(image)
    indexConcave, marksCont = pulisciContorni(contorni)
    cv2.drawContours(image, contours=marksCont, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    objPoints,imgPoints = estraiPunti(marksCont, indexConcave, image)
    ret, rvecs, tvecs = cv2.solvePnP(objPoints, imgPoints, mtx, dist, cv2.SOLVEPNP_IPPE)

    """ projImgPts, jac = cv2.projectPoints(objPoints, rvecs, tvecs, mtx, dist)
    xErr, yErr = calcolaRMS(projImgPts, imgPoints)
    if xErr>1 or yErr>1:
        print("RMS maggiore di 1! (" + str(xErr) + ", " + str(yErr) + ")") """

    return rvecs, tvecs

"""
Funzione che dato il numero di voxel desiderati, la misura del lato del cubo grande (formato dall'insieme dei voxel)
e la posizione dell'angolo in alto a sinistra della faccia frontale del cubo, resituisce la posizione del centro di ogni voxel.

I centri vengono calcolati dalla facciata frontale, andando in profondità, per ogni facciata partendo dalla riga più in alto,
da sinistra verso destra.
"""
def getVoxelsCenters(nVoxels, side, topLeft):
    centers = np.zeros((nVoxels**3,3), np.float32)
    shift = side/nVoxels                #lato di un voxel
    for i in range(nVoxels):            #facciata
        for j in range(nVoxels):        #riga
            for k in range(nVoxels):    #colonna
                index = i*nVoxels**2 + j*nVoxels + k
                cx = topLeft[0] - i*shift - shift/2
                cy = topLeft[1] + k*shift + shift/2
                cz = topLeft[2] - j*shift - shift/2
                centers[index] = np.float32([cx,cy,cz])
    return centers
                
"""
Funzione che dato un frame "scolpisce" i voxel rimuovendo quelli non compresi nella silhouette dell'oggetto.
    -image: frame corrente
    -voxelCenters: centri dei voxel parte dell'oggetto fino a questo punto
    -mtx, dist: parametri ottenuti dalla calibrazione della camera
    -DRAW: se True mostra a schermo i voxel attualmente validi, sovrapposti al frame mostrato a video.
"""
def carve(image, voxelCenters, mtx, dist, DRAW):
    rvecs, tvecs = poseEstimation(image, mtx, dist)
    sil = estraiSilhouette(image)
    imgPts, jac = cv2.projectPoints(voxelCenters, rvecs, tvecs, mtx, dist)      #proiezione dei centri sull'immagine
    okPts = []

    for p, point in enumerate(imgPts):
        x = int(point[0][0])
        y = int(point[0][1])
        if y<1080 and x<1920 and sil[y][x] != 0:    #il pixel relativo a questo centro non è nero -> parte dell'oggetto
            okPts.append(voxelCenters[p])
            if DRAW:
                cv2.drawMarker(image, (x, y),(0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1, line_type=cv2.LINE_AA)
    okPts = np.array(okPts)
    cv2.imshow("borders", image)
    return okPts

"""
Funzione che salva i centri dei voxel in un file .ply per poter essere visualizzati con un software esterno, quale MeshLab.
""" 
def saveToPLY(name, voxelCenters):
    nPoints = len(voxelCenters)
    fullName = "results" + name + ".ply"
    f = open(fullName, "w")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex " + str(nPoints) +"\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")
    for vc in voxelCenters:
        f.write(str(vc[0]) + " " + str(vc[1]) + " " + str(vc[2]) + "\n")
    f.close()


# +++++ INIZIO PARTE DI TEST +++++
obj = int(input("Inserisci numero del file da scolpire (1-4)"))
if obj >0 and obj<5:
    nomeFile = "\obj0"+str(obj)
else:
    print("Oggetto non trovato, obj01 default")
    nomeFile = "\obj01" #default
vidcap = cv2.VideoCapture("data" + nomeFile + ".mp4")

#si assume che precedentemente sia stata calibrata la camera
with open('mtx.pkl', 'rb') as f:
    mtx = pickle.load(f)
with open('dist.pkl', 'rb') as f:
    dist = pickle.load(f)


nVox = 70               #numero di voxel per lato
side = 130              #lato del cubo
up = 80                 #altezza del cubo rispetto al piatto
cubeTopLeftCorner = np.float32([+side/2,-side/2, side+up]) #coordinate angolo in alto a sinistra del cubo grande (facciata frontale)
voxelCenters = getVoxelsCenters(nVox, side, cubeTopLeftCorner)
frameCounter = 0

print("Inizio carving..")
while True:    
    ret, image = vidcap.read()
    frameCounter += 1
    if not ret:
        break

    if frameCounter%10 == 0:    #operazione effettuata ogni 10 frame
        voxelCenters = carve(image, voxelCenters, mtx, dist, DRAW=True)
    k = cv2.waitKey(1) & 0xff    
    #Se viene premuta q, il programma si interrompe
    if k == ord('q'):
        break

# Release the VideoCapture Object.
print(".. fine carving!")
vidcap.release()
if not os.path.exists("results"):
    os.makedirs("results")
saveToPLY(nomeFile, voxelCenters)
cv2.destroyAllWindows()
