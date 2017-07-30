import os.path
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from sklearn.utils import shuffle
from random import randint
import random

samples = []
with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        samples.append(line)
#
# with open('./data/first_reentry/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader, None)  # skip the headers
#     for line in reader:
#         samples.append(line)

# with open('./data/personal_rec2/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader, None)  # skip the headers
#     for line in reader:
#         samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img_id = randint(0, 2)

                if '/home/' not in batch_sample[0]:
                  name = './data/data/IMG/'+batch_sample[img_id].split('/')[-1]
                else: # sample_set == '2'
                  name = './data/first_reentry/IMG/'+batch_sample[img_id].split('/')[-1]

                if not os.path.isfile(name):
                  print("file {} does not exist".format(name))

                image = cv2.imread(name)
                angle = float(batch_sample[3])
                side_cam_offset = 0.2
                if img_id == 1:
                  angle += side_cam_offset
                if img_id == 2:
                  angle -= side_cam_offset

                if (bool(random.getrandbits(1))):
                  images.append(image)
                  angles.append(angle)
                else:
                  images.append(np.fliplr(image))
                  angles.append(-angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 90, 320  # Trimmed image format

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=10)
model.save('model.h5')