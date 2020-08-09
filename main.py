import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout,PReLU, GlobalAveragePooling2D
from tensorflow.python.keras.saving.saved_model.load import metrics

df = pd.read_csv('train_data.csv')
train_images = []
for i in df[df.columns[0]]:
    im = cv2.imread('train/'+i,cv2.IMREAD_GRAYSCALE)
    train_images.append(im)
images = []
for pixel_sequence in train_images:
    im = cv2.resize(pixel_sequence.astype('uint8'), (32,32))
    images.append(im.astype('float32'))
images = np.asarray(images)
images = np.expand_dims(images, -1)
train_labels = df[df.columns[2]]
x_train,x_test,y_train,y_test = train_test_split(images,train_labels,test_size = 0.30)

input_s = (32,32,1)
input = Input(input_s)
x = Conv2D(128,(5,5),strides= (1,1),padding = 'same',activation='relu')(input)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(96)(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
x = Dense(54)(x)
x = PReLU(alpha_initializer='zeros',alpha_constraint=None,alpha_regularizer=None)(x)
x = Dropout(0.25)(x)
x = Dense(1)(x)
output = Activation('sigmoid')(x)

model = Model(input,output)
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
y_pred = model.predict(x_test,10)
print(model.summary())
history = model.fit(x_train, y_train,validation_split = 0.30, epochs=50, batch_size=100)
his_dic = history.history
print(his_dic.keys())
score, accu = model.evaluate(x_test,y_test)
classes = ['0','1']
print('Test Score:',score)
print('Test accuracy:',accu)
#model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# model.fit(x_train,y_train,batch_size = 100,epochs = 15,verbose = 1,validation_data=(x_test,y_test))
#loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('losss')
plt.xlabel('epoch')
plt.legend('train',loc='upper left')
plt.show()
#confusion matrix

print("*********************************")
print(np.asarray(y_test).argmax(axis=1))
print(y_pred.argmax(axis=1))
confusion_matrix = metrics.confusion_matrix(np.asarray(y_test).argmax(axis = 1),y_pred.argmax(axis=1))
print(confusion_matrix)
plt.imshow(confusion_matrix, interpolation="nearest",cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.show()
