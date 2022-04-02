import glob 
import cv2
import sys
import os
filename=r'C:\Users\.....path…….\surprise' 
os.mkdir('Cropped’)                     # To create  a new folder
path=r’C:\....path…..\Cropped'
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')     #First download Haarcascades xml file 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces
i=0
for img in glob.glob(filename+'/*.*'):
    var_img = cv2.imread(img)
    face = detect_face(var_img)
    print(face)
    if (len(face) == 0):
        continue
    for(ex, ey, ew, eh) in face:
        crop_image = var_img[ey:ey+eh, ex:ex+ew]
        cv2.imwrite(os.path.join(path,'new%04i.jpg' %i),crop_image)
        cv2.imshow("cropped", crop_image)
        cv2.waitKey(20) 
    i=i+1
cv2.destroyAllWindows()
