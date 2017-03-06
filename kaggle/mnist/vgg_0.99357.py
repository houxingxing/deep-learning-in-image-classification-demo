import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,BatchNormalization, ZeroPadding2D
import keras.models as models
import keras.utils.np_utils as kutils
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from keras.datasets import mnist 
train = pd.read_csv("C:\\Users\\Administrator\\Documents\\sublime\\train.csv").values
test  = pd.read_csv("C:\\Users\\Administrator\\Documents\\sublime\\test.csv").values
X_train,y_train,X_test,y_test=mnist.load()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


nb_epoch = 15 # Change to 100

batch_size = 128
img_rows, img_cols = 28, 28

nb_filters_1 = 32 # 64
nb_filters_2 = 64 # 128
nb_filters_3 = 128 # 256
nb_filters_4 = 256 # 256
nb_conv = 3

trainX = train[:, 1:].reshape(train.shape[0],1, img_rows, img_cols)
trainX = trainX.astype(float)
trainX /= 255.0

trainY = kutils.to_categorical(train[:, 0])
nb_classes = trainY.shape[1]

cnn = models.Sequential()
cnn.add(ZeroPadding2D((1,1),input_shape=(1,28,28)))
cnn.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, border_mode='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Convolution2D(nb_filters_1, nb_conv, nb_conv, border_mode='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D((2,2),strides=(2, 2)))



#cnn.add(ZeroPadding2D((1,1)))
cnn.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, border_mode='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Convolution2D(nb_filters_2, nb_conv, nb_conv, border_mode='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D((2,2),strides=(2, 2)))




#cnn.add(ZeroPadding2D((1,1)))
cnn.add(Convolution2D(nb_filters_3, nb_conv, nb_conv, border_mode='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Convolution2D(nb_filters_3, nb_conv, nb_conv, border_mode='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D((2,2),strides=(2, 2)))



#cnn.add(ZeroPadding2D((1,1)))
#cnn.add(Convolution2D(nb_filters_4, nb_conv, nb_conv, border_mode='same'))
#cnn.add(BatchNormalization())
#cnn.add(Activation('relu'))
#cnn.add(ZeroPadding2D((1,1)))
#cnn.add(Convolution2D(nb_filters_4, nb_conv, nb_conv, border_mode='same'))
#cnn.add(BatchNormalization())
#cnn.add(Activation('relu'))
#cnn.add(ZeroPadding2D((1,1)))
#cnn.add(MaxPooling2D((2,2),strides=(2, 2)))

cnn.add(Dropout(0.5))
cnn.add(Flatten())
cnn.add(Dense(512 , activation="relu")) # 4096
cnn.add(Dense(128, activation="relu")) # 4096
cnn.add(Dense(nb_classes, activation="softmax"))

cnn.summary()
cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

es=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='max')
checkpointer = ModelCheckpoint(filepath="mnist_weights-origin.hdf5", verbose=1, save_best_only=True)

cnn.load_weights('mnist_weights-origin.hdf5')
history=cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,validation_split=0.1,callbacks=[es,checkpointer])

testX = test.reshape(test.shape[0],1, 28, 28)
testX = testX.astype(float)
testX /= 255.0

yPred = cnn.predict_classes(testX)

plt.figure
plt.plot(history.epoch,history.history['acc'],label="acc")
plt.plot(history.epoch,history.history['val_acc'],label="val_acc")
plt.scatter(history.epoch,history.history['acc'],marker='*')
plt.scatter(history.epoch,history.history['val_acc'])
plt.legend(loc='under right')
plt.show()

plt.figure
plt.plot(history.epoch,history.history['loss'],label="loss")
plt.plot(history.epoch,history.history['val_loss'],label="val_loss")
plt.scatter(history.epoch,history.history['loss'],marker='*')
plt.scatter(history.epoch,history.history['val_loss'],marker='*')
plt.legend(loc='upper right')
plt.show()
np.savetxt('mnist-vggnet.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')