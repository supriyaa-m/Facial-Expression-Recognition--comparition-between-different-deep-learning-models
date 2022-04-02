import cv2
import glob
import os
import numpy as np
Directory=r'C:\.....path…\sad'  #Have to change folder names accordingly
os.mkdir('New’)            To create new folder
path=r'C:\....path….\New'
i=0
for img in glob.glob(Directory + "/*.jpg"):
    image=cv2.imread(img)
    resized_img = cv2.resize(image, (75 , 75))
    norm_image = cv2.normalize(resized_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image.astype(np.uint8)
    cv2.imwrite(os.path.join(path,'new%04i.jpg' %i),norm_image)        #To save in the new folder created, can give any name
    i=i+1
    cv2.imshow('image',resized_img)
    cv2.waitKey(30)
cv2.destroyAllWindows()
