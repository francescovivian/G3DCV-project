import cv2
import os
import shutil
import numpy as np
import glob
import pickle

desiredFrames=20
vidcap = cv2.VideoCapture("data\calibration.mp4")
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) #numero totale di frame nel video

#calcolo quanti frame saltare tra un salvataggio e l'altro
step = int(frame_count/desiredFrames)

#estrazione dei frame dal video
if not os.path.exists("frames"):
    os.makedirs("frames")
for i in range(0,desiredFrames):
    vidcap.set(1, i*step)
    ret, frame = vidcap.read()
    cv2.imwrite(r"frames\frame%d.jpg" %i, frame)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6,3), np.float32)  
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
objpoints = [] # punti 3d nel "mondo reale"
imgpoints = [] # punti 2d nel piano dell'immagine
images = glob.glob(r"frames\*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6,9), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        #Disegna gli angoli nell'immagine
        cv2.drawChessboardCorners(img, (6,9), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)
cv2.destroyAllWindows()

print("Inizio camera calibration..")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#salvataggio di mtx e dist in due file separati
afile = open('mtx.pkl', 'wb')
pickle.dump(mtx, afile)
afile.close()
afile = open('dist.pkl', 'wb')
pickle.dump(dist, afile)
afile.close()

shutil.rmtree("frames") #rimuove la cartella con i frame e i file all'interno

print(".. termine camera calibration, salvataggio effettuato")