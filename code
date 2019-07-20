import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
from keras.preprocessing .image import ImageDataGenerator,img_to_array,load_img
from os import listdir
from os.path import isfile, join
import cv2
import xml.etree.ElementTree as ET
from keras import applications
from keras.models import Model, load_model
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.layers import Input, Add,Dropout, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras.applications.imagenet_utils import preprocess_input

a=''
b=''

y=[]
mypath=a
X=[]
onlyfile = [ f for f in listdir(b) if isfile(join(b,f))]
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
for n in range(0, len(onlyfiles)):
    images = cv2.imread( join(mypath,onlyfiles[n]),cv2.IMREAD_COLOR )
    images = cv2.resize(images, (64,64),interpolation=cv2.INTER_CUBIC)
    c=onlyfiles[n]
    X.append(images)
    doc = ET.parse(join(b,onlyfile[n]))
    for i in doc.findall("filename"):
        if c==i.text:
            d=[]
            for j in doc.findall("object/name"):
                d.append(j.text)
            y.append(d)
            d=[]
    #print(images.shape)
    #plt.imshow(images)
    
X=np.array(X)
X=X/255

df=pd.DataFrame()
df['col']=y
df.head()
yy=[list(dict.fromkeys(c)) for c in df.col]
df['loc']=yy
df['y']=df['loc'].apply(', '.join)
df.head()
df.y.value_counts()

df.y.replace(['Dislocation','Large_tear_or_damage','Tear','Dent, Dislocation'],['Dent','Dent','Dent','Dent'],inplace=True)
df.y.replace(['Dislocation, Dent','Tear, Scratch_or_spot','Shatter'],['Scratch_or_spot','Scratch_or_spot','Scratch_or_spot'],inplace=True)
df.y.replace(['Dislocation','Large_tear_or_damage','Tear','Shatter','Dent, Dislocation'],['Dent','Dent','Dent','Dent','Dent'],inplace=True)
df.y.replace(['Tear, Dent','Dent, Tear','Dislocation, Dent, Scratch_or_spot','Dislocation, Tear'],['Dent','Dent','Dent','Dent'],inplace=True)
df.y.replace(['Dent, Scratch_or_spot','Dislocation, Scratch_or_spot'],['Scratch_or_spot','Scratch_or_spot'],inplace=True)
df.y.replace(['Scratch_or_spot, Dent','Scratch_or_spot, Dislocation','Scratch_or_spot, Dislocation'],['Scratch_or_spot','Scratch_or_spot','Scratch_or_spot'],inplace=True)
df.y.replace(['Tear, Dislocation','Large_dent','Large_dent, Tear','Dislocation, Scratch_or_spot, Dent'],['Dent','Dent','Dent','Dent'],inplace=True)
df.y.replace(['Scratch_or_spot, Tear','Large_tear_or_damage, Dent','Dent, Shatter'],['Scratch_or_spot','Dent','Dent'],inplace=True)
df.y.replace(['Dislocation, Tear, Dent','Large_tear_or_damage, Dent, Tear','Dent, Large_tear_or_damage','Tear, Shatter'],['Dent','Dent','Dent','Dent'],inplace=True)
df.y.replace(['Tear, Large_tear_or_damage','Dislocation, Scratch_or_spot, Tear','Tear, Dent, Scratch_or_spot','Dislocation, Tear, Dent, Scratch_or_spot'],['Dent','Dent','Dent','Dent'],inplace=True)
df.y.replace(['Scratch_or_spot, Tear, Dent','Large_tear_or_damage, Shatter','Dislocation, Scratch_or_spot, Dent, Tear'],['Dent','Dent','Dent'],inplace=True)
df.y.replace(['Large_tear_or_damage, Dislocation','Dent, Dislocation, Scratch_or_spot'],['Dent','Dent'],inplace=True)


df.y.value_counts()
df['y']=df.y.apply(lambda x: 0 if x=="Dent" else 1)
df.head()

print(df.y.value_counts())
df.dtypes
test=df.y
test.shape

tst=np.array(test)
tst.shape
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val=train_test_split(X,tst,test_size=.1,random_state=2019)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
ntrain=len(X_train)
nval=len(X_val)
batch_size=32
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
y_train = convert_to_one_hot(y_train, 2).T
y_val = convert_to_one_hot(y_val, 2).T

print(y_val.shape)
img_height,img_width = 64,64
num_classes = 2

base_model = applications.resnet50.ResNet50( include_top=False, weights='imagenet',input_shape= (img_height,img_width,3))
x = base_model.output
x = Flatten()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512,activation='relu')(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
train_dat=ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)
val_dat=ImageDataGenerator()

train_gen=train_dat.flow(X_train,y_train,batch_size = 32)

val_gen=val_dat.flow(X_val,y_val,batch_size = 32)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=.1)
 #early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
#checkpoint = ModelCheckpoint("Restnet50.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.fit_generator(train_gen,steps_per_epoch=ntrain//32,epochs=18,
                         validation_data=val_gen,validation_steps=ntrain//32,
                        callbacks=[learning_rate_reduction])
pred = model.evaluate(X_train, y_train)
print ("Loss = " + str(pred[0]))
print ("Train Accuracy = " + str(pred[1]))
pred = model.evaluate(X_val, y_val)
print ("Loss = " + str(pred[0]))
print ("Test Accuracy = " + str(pred[1]))
p=model.predict(X_val, batch_size=16, verbose=0, steps=None)

fig, axs = plt.subplots(8,7, figsize=(25, 46), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()
for i in range(len(X_val)):
    axs[i].imshow(X_val[i])
    axs[i].set_title(np.argmax(p[i]))
plt.show()
