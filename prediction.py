from keras.models import load_model         
model=load_model('XcepResnetM1.hdf5’)                                   #load the model saved before
loss_and_metrics = model.evaluate(validation_dataset)                #check accuracy for validation set
print(loss_and_metrics)

img = cv2.imread('happy.jpg’)                                                          #input an image
img = cv2.resize(img,(75,75),3)                                                        #resize the image according to the input size of model
test_img=img.reshape((-1,75,75,3))
test_img=test_img/255.0

img_class = model.predict_classes(test_img)                                       #predict the label
prediction = img_class[0]
classname = img_class[0]
print("Class: ",classname)

img =cv2.resize(img,(224,224),3)                                                      #to display the image along with its label predicted
plt.imshow(img)
plt.title(classname)
plt.show()



