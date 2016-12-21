from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D 
from keras.optimizers import SGD
from keras.layers import GRU, TimeDistributed, RepeatVector, Merge, TimeDistributedDense
import h5py
import text_processor_utils as tp_utils
import cv2
import numpy as np
from keras.preprocessing import sequence

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('imclass_data/train/cats/cat.0.jpg')
x = img_to_array(img)
print x.shape
x = x.reshape((1,)+x.shape)
j=0
for batch in datagen.flow(x, batch_size=1,save_to_dir='preview',save_prefix='cat', save_format='jpeg'):
  j+=1
  if(j>20):
    break;
model = Sequential()
model.add(Convolution2D(32,3,3,input_shape=(3,150,150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1/.255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1/.255)


train_generator = train_datagen.flow_from_directory('imclass_data/train',target_size=(150,150), batch_size=32, class_mode="binary")
validation_generator = test_datagen.flow_from_directory('imclass_data/train',target_size=(150,150), batch_size=32, class_mode="binary")


model.fit_generator(train_generator, samples_per_epoch=2000, nb_epoch=50, validation_data=validation_generator,nb_val_samples=800)
model.save_weights('first_try.h5')

