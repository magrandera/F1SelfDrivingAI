from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import rmsprop
from keras.layers.normalization import BatchNormalization

def maiav1():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(133, 180, 3), strides=4, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(3,3), strides=4))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))
    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))
    # Step 3 - Flattening
    classifier.add(Flatten())
    classifier.add(Dense(512, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=9, activation='softmax'))
    opt = rmsprop(lr=0.0001, decay=1e-6)
    # Compiling the CNN
    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier