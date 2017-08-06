import os.path
import csv
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from sklearn.utils import shuffle
import cv2
import numpy as np
from random import randint
import random

# all lines from csv files are sored in this object
samples = []

# image preprocessing:
# applies cropping, gaussian blur, BGR to YUV conversion and resize to size proposed by NVidia
# (conversion in YUV as proposed in their blog "End-to-End Deep Learning for Self-Driving Cars"
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
def preprocess_image(image_file_name):
  image = cv2.imread(image_file_name)
  cropped_img = image[50:140, :, :]
  cropped_img = cv2.GaussianBlur(cropped_img, (3, 3), 0)
  cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2YUV)
  return cv2.resize(cropped_img, (200, 66), interpolation=cv2.INTER_AREA)


# helper to read csv files
#  - uses all 3 images (left, right, center) randomly choosing one of the 3 (not all images are used)
#  - applies an offset to steering angle when using left or right image
#  - flips images randomly and negates the angle if image was flipped
def read_all_samples(csv_file):
  print("reading samples from file...")
  reader = csv.reader(csv_file)
  next(reader, None)  # skip the headers
  for line in reader:
    if float(line[6]) >= 0.1:
      img_id = randint(0, 2)
      if '/home/' not in line[0]:
        img_name = './data/data/IMG/'+line[img_id].split('/')[-1]
      else: # personal sets
        res = line[img_id].split('/')
        img_name = './data/' + res[-3] + '/' + res[-2] + '/' + res[-1]

      if not os.path.isfile(img_name):
        print("file {} does not exist".format(img_name))
        exit(-1)

      image = preprocess_image(img_name)
      angle = float(line[3])

      side_cam_offset = 0.25
      if img_id == 1:
        angle += side_cam_offset
      elif img_id == 2:
        angle -= side_cam_offset

      if (bool(random.getrandbits(1))):
        samples.append([np.fliplr(image), -angle])
      else:
        samples.append([image, angle])


### first track files
# Udacity samples
with open('./data/data/driving_log.csv') as udacity_csvfile:
  read_all_samples(udacity_csvfile)

# forward driving samples
with open('./data/personal_rec4/driving_log.csv') as rec4_csvfile:
  read_all_samples(rec4_csvfile)

# reverse driving samples
with open('./data/first_reverse_rec5/driving_log.csv') as rec5_csvfile:
  read_all_samples(rec5_csvfile)

# forward recovery driving samples
with open('./data/first_reentry/driving_log.csv') as reentry_csvfile:
  read_all_samples(reentry_csvfile)

# reverse driving recovery samples 1
with open('./data/first_reverse_reentry/driving_log.csv') as reentry_reverse_csvfile:
  read_all_samples(reentry_reverse_csvfile)

# reverse driving recovery samples 2
with open('./data/first_extension/driving_log.csv') as extension_csvfile:
  read_all_samples(extension_csvfile)

### second track files
# forward driving samples
with open('./data/personal_rec2/driving_log.csv') as rec2_csvfile:
  read_all_samples(rec2_csvfile)

# forward driving samples 2
with open('./data/personal_rec3/driving_log.csv') as rec3_csvfile:
  read_all_samples(rec3_csvfile)

# recovery rving samples
with open('./data/second_reentry/driving_log.csv') as second_reentry_csvfile:
  read_all_samples(second_reentry_csvfile)

# splitting samples into training and validation sets here
shuffle(samples)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.02)

# generator that shuffles images (sub_samples):
def generator(sub_samples, batch_size=32):
    num_samples = len(sub_samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle(sub_samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = sub_samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
              images.append(batch_sample[0])
              angles.append(batch_sample[1])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Trimmed image format, same as NVidia
ch, row, col = 3, 66, 200

# Preprocessing
model = Sequential()
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
# Building network
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
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
# summary
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=10)
# saving the model
model.save('model.h5')
