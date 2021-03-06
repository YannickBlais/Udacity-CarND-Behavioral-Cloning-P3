# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use provided images and the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/network_model.png "Model Visualization"
[image2]: ./examples/center_lane.jpg "Center lane example"
[image3]: ./examples/recovery_1.jpg "Recovery Image - far"
[image4]: ./examples/recovery2.jpg "Recovery Image - mid"
[image5]: ./examples/recovery3.jpg "Recovery Image - center"
[image6]: ./examples/normal_image.jpg "Normal Image"
[image7]: ./examples/flipped_image.jpg "Flipped Image"
[image8]: ./examples/left_camera.jpg "Left Camera Image"
[image9]: ./examples/center_camera.jpg "Center Camera Image"
[image10]: ./examples/right_camera.jpg "Right Camera Image"
[image11]: ./examples/original_image.png "Original Image (without preprocessing)"
[image12]: ./examples/preprocessed_image.png "Preprocessed Image"

[video1]: ./videos/first_track_9.mp4 "First Track, 9mph"
[video2]: ./videos/first_track.mp4 "First Track, 15mph"
[video3]: ./videos/second_track_9.mp4 "Second Track, 9mph"
[video4]: ./videos/second_track.mp4 "Second Track, 15mph"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md (this file) summarizing the results
* examples folder containing images used in this report
* videos folder containing videos produced of the car autonomously driving on both tracks

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network modeled on the Nvidia CNN used for End-to-End driverless car. It consists of 5 convolutional layers where the first 3 layers use 2x2 pooling and 5x5 filters with outputs ranging from 24 to 48 and the last 2 use a 3x3 filter with both outputs of 64 (model.py lines 140-144).

The model includes RELU layers to introduce nonlinearity on all 5 convolutional layers (model.py lines 140-144), and the data is normalized in the model using a Keras lambda layer (model.py lines 136-138).

Not included in keras model but performed as preprocessing are the following tasks:
- Cropping to remove top 50 and bottom 20 lines
- Gaussin blur to reduce noise
- Conversion to YUV color format as decribed in NVidia [blog](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
- Resizing image to 200x66 (also used by NVidia) using a simple but fast openCV INTER_AREA method

#### 2. Attempts to reduce overfitting in the model

The model fully convolutional layers contain dropout layers in order to reduce overfitting (model.py lines 146, 148, 150 and  152). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 103). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 156).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the track in reverse as well as recordings from both tracks. All the various driving samples are loaded and described in model.py lines 62 to 98.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a networks that is already proven and modern so I chose to use right from the beginning the model proposed by NVidia in their [blog](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). This network is appropriate because it uses multiple convolutional layers which is relevant in our case because we have images as inputs. Following the convolutional layers are multiple fully connected layers which is the standard approach and it ends with a single floating point value which is all that what we need to model our steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The splitting is performed randomly at the start of the program and 20% of samples are reserved for validation. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that it uses dropout with the 4 fully convolutional layers. All dropout uses a keep_prob of 0.5 for training.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Too improve the driving behavior in these cases, I recorded more recovery driving from those locations. During my preprocessing development, I had some issues with the autonomous driving as I later found out that cv2.imread loads images in BGR format while the simluator returns images in RGB format. I also realized that even if there is a little improvement from the loss function, more training epochs helped improve driving.

When the vehicle could perform well on the first track I recorded samples on the second track and performed the same steps of  evaluating autonomous driving/adding recovery driving where necessary.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road, even at 15mph.

#### 2. Final Model Architecture

The final model architecture (model.py lines 140-153) consisted of a convolution neural network with the following layers and layer sizes (using model.summary() method from Keras):
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1152)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_3[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_4[0][0]                  
====================================================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
____________________________________________________________________________________________________
```

Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to the center of the road if/when it strays away from it. These images show what a recovery looks like starting from the edge of the road, bringing the car back to the center :

![alt text][image3]
![alt text][image4]
![alt text][image5]

In order to get my model to generalize better, I also recorded images while driving on the track in reverse. Likewise, I recorded images of the vehice recovering from left and right to the center while driving in reverse.

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles (see model.py lines 56-59) thinking that this would help generalize the training. For example, here is an example of an image and its flipped counterpart:

![alt text][image6]
![alt text][image7]

I also included the left and right images. In order to do so I had to apply a small correction offset to the angle applied to those images as described in the course. After some tests, I used a correction of 0.25 that seemed to give the best driving results. Using the left and right images helped generalized the training by augmenting the input data. Here are some examples of left and right images:

![alt text][image8]
![alt text][image9]
![alt text][image10]

After the collection process, I had 30552 number of data points (each data point includes 3 images: left, right and center). I then preprocessed this data by first cropping it and removing the top 50 pixels and the bottom 20. This removes the unnecessary part of the picture that contains non-relevant information such as scenery or the hood of the car. I then apply a simple 3x3 guassian blur to reduce noise, convert to YUV color space as suggested by NVidia and resized the image to 200x66. I also perform normalizing of the images by diving the pixel values by 127.5 and subtracting 1.0 to bring them between -1.0 nd 1.0 which makes the model normally easier to converge.

Here is an example of an image pre- and post-preprocessing:

![alt text][image11]
![alt text][image12]

I randomly shuffled the data set and put 20% of the data into a validation.

I used the remaining 80% as training data for the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs I found was 10 as evidenced by the good driving of the vehicle on both tracks. More itereations reduced slightly the training and validation loss but did not improve the actual driving. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The final videos can be found here:
- [First track @9mph][video1]
- [First track @15mph][video2]
- [Second track @9mph][video3]
- [Second track @15mph][video4]
