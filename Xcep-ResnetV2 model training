import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.optimizers import Adam,RMSprop,SGD,Nadam
from keras.models import Sequential, Model
from keras.applications import ResNet50V2
from keras.applications.xception import Xception
#from keras.layers.core import Flatten, Dense, Dropout,Activation
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D,Input,concatenate
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomNormal
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from keras.utils import np_utils
import scikitplot
from keras.models import model_from_json
import math

#Note: You have to first manually install scikit_plot in ur conda environment -> open anaconda prompt-> open new env created using command 
# "conda activate env_name" -> then to install -> "conda install -c conda-forge scikit-plot".

INPUT_PATH ="C:/.....path.../Final_Data5/"
total_images = 0
for dir_ in os.listdir(INPUT_PATH):
    count = 0
    for f in os.listdir(INPUT_PATH + dir_ + "/"):
        count += 1
        total_images += 1
    print(f"{dir_} has {count} number of images")
 print(f"\ntotal images are {total_images}")
 
 TOP_EMOTIONS = ['Angry','Disgust','Fear','Happy','Normal','Sad','Surprise']
img_arr = np.empty(shape=(total_images,75,75,3))
img_label = np.empty(shape=(total_images))
label_to_text = {}
i = 0
e = 0
for dir_ in os.listdir(INPUT_PATH):
    if dir_ in TOP_EMOTIONS:
        label_to_text[e] = dir_
        for f in os.listdir(INPUT_PATH + dir_ + "/"):
            img_arr[i] = cv2.imread(INPUT_PATH + dir_ + "/" + f)
            img_label[i] = e
            i += 1
        print(f"loaded all {dir_} images to numpy arrays")
        e += 1

img_arr.shape, img_label

idx =-1
for k in label_to_text:
    sample_indices = np.random.choice(np.where(img_label==k)[0], size=7, replace=False)
    sample_images = img_arr[sample_indices]
    for img in sample_images:
        idx += 1
        ax = plt.subplot(7,7,idx+1)
        ax.imshow(img[:,:,0], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label_to_text[k])
        plt.tight_layout()

img_label = np_utils.to_categorical(img_label)
img_label.shape


X_train, X_test, y_train, y_test = train_test_split(img_arr, img_label,shuffle=True, stratify=img_label,\
                                                    train_size=0.8, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


full_name='concatenate'                                         #creating a layer name
input_tensor=Input(shape=(75,75,3))                             #Input shape
base_model1 = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)   #extracting best weights from the model trained using imagenet dataset
features1 = base_model1.output                                                             #storing features in features1 variable
base_model2 = ResNet50V2(weights='imagenet', include_top=False, input_tensor=input_tensor) #Same procedure for 2nd model
features2 = base_model2.output


concatenated=concatenate([features1,features2])            #Concatenate the extracted features

conv=Conv2D(1024,(1, 1),padding='same')(concatenated)      #add the concatenated features to a convolutional layer
feature = Flatten(name='flatten')(conv)
dp = Dropout(0.5)(feature)                                 #add dropout
preds = Dense(7, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp) 
Concatenated_model = Model(inputs=input_tensor, outputs=preds)

for layer in Concatenated_model.layers:                 
  layer.trainable = True

learning_rate = 0.0001
opt =Nadam(lr=learning_rate)                            #This is adam with Nestrov momentum , its the best optimizer for this model
Concatenated_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

filepath=filepath="XcepResnetM1.hdf5"
reduce_lr=ReduceLROnPlateau(monitor='val_accuracy',factor=0.2,patience=5,min_lr=1e-8,verbose=1,model='auto')
early_stopping_cb = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=35,restore_best_weights=True)   
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',verbose=1,save_best_only=True, mode='max')                    

train_datagen = ImageDataGenerator(
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    zca_whitening=False,
)
train_datagen.fit(X_train)

callbacks=[reduce_lr,early_stopping_cb,checkpoint]
batch_size=32
epochs=100

history = Concatenated_model.fit_generator(
    train_datagen.flow(X_train, y_train,batch_size=batch_size),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train)/batch_size,
    shuffle=True, 
    epochs=epochs,
    callbacks=callbacks,
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model_json = model.to_json()
with open("XcepResnetM1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("XcepResnetM1.h5")   #weights saving

#Plotting confusion matrix

label_to_text
{0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Normal',5:'Sad',6:'Surprise'}
text_to_label = dict((v,k) for k,v in label_to_text.items())
text_to_label
{'Angry': 0,'Disgust':1,'Fear':2,'Happy':3,'Normal':4 ,'Sad':5,'Surprise':6 }
yhat_test = np.argmax(model.predict(X_test), axis=1)
ytest_ = np.argmax(y_test, axis=1)

scikitplot.metrics.plot_confusion_matrix(ytest_, yhat_test, figsize=(7,7))
plt.savefig("confusion_matrix_modelname.jpg")   #saving ur confusion matrix plot image in drive

test_accu = np.sum(ytest_ == yhat_test) / len(ytest_) * 100
print(f"test accuracy: {round(test_accu, 4)} %\n\n")

print(classification_report(ytest_, yhat_test))


