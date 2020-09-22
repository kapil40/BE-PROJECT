from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes = 3             # the three target class
img_rows,img_cols = 48,48   # the size of the training image
batch_size = 32             # processing 32 images at a time in a batch

train_data_dir = r"C:\Users\KAPIL\Downloads\fer2013-20200119T102918Z-001\fer2013\train"
validation_data_dir = r'C:\Users\KAPIL\Downloads\fer2013-20200119T102918Z-001\fer2013\validation'

# Use to generate multiple images from a single image in different forms specified
train_datagen = ImageDataGenerator(
					rescale=1./255,      # To normalize the image
					rotation_range=30,   # Rotates the image left and right by 30 degree
					shear_range=0.3,
					zoom_range=0.3,      # Zooms the image by 30 percent
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,  # Flip the image horizontally
					fill_mode='nearest')  # Used to avoid loss of pixels

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,           # Defined in the directory
					color_mode='grayscale',   # We are working on grey scale
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical', # Since there are nore than two classes to classified so we use categorical here
					shuffle=True)             # We want to shuffle the data to ensure that model does not know anything beforehand

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


model = Sequential()

# Block-1-The convolutional layer

# The kernel will be of size 3*3, there will be 32 kernels  
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))        #We use elu function here beacause it works well for classification for more than 2 classes and it's computational rate is faster
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # Used to reduce the size of image dimensions and take the maximum value
model.add(Dropout(0.2)) # As each neuron can carry the data there are chances of overfitting of data so now 20% of neurons will be switched of in random manner 
# It will distribute the weight of neuron evenly
# Block-2 

# Here there are 64 kernels of size 3*3 and here we do not give input as we use output of previous block 
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

# Here there are 128 kernels of size 3*3 and here we do not give input as we use output of previous block
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))      #Kernel initializer is used to allocate random weights
model.add(Activation('elu'))
model.add(BatchNormalization())                                                 #It is used to adjust the mean so that there is no value loss
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

# Here there are 256 kernels of size 3*3 and here we do not give input as we use output of previous block
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5-The fully connected layer

model.add(Flatten())    # It will convert all the matrix data(i.e 2-D data) into one dimensional data
model.add(Dense(64,kernel_initializer='he_normal'))   # We use 64 kernels in this dense layer
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))   # Only 50% of neuron will be activated

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

# We only need 3 classes in the end so we now define kernel size of 3 which is equivalent to no of target class
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                             monitor='val_loss',  # To monitor validation loss to see whether the loss is decreasing or not
                             mode='min',
                             save_best_only=True, # Saves best model with minimum validaion loss 
                             verbose=1)           # Use to display the details about the model

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,   # If value is less than min_delta then it will stop 
                          patience=9,    # Check till 9 epochs if no improvement then stop training 
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,  # Decrease the learning rate by the factor of 0.2
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 17060
nb_validation_samples = 1996
epochs=25

# Used to train the model
history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)























































