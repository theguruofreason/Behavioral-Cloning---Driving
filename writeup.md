#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_data_4/IMG/center_2017_04_23_22_21_24_258.jpg "Center Lane Driving"
[image2]: ./normal.jpg "Normal Image"
[image3]: ./flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

#1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* cnn.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#3. Submission code is usable and readable

The cnn.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

#1. An appropriate model architecture has been employed

My model employs a lambda layer to normalize the data (cnn.py line 41), followed by a 2D cropping layer to cut out the sky (cnn.py line 42). Then it has 3 5x5 2D convolutional layers each followed by a relu activation and then max pooling 2D (cnn.py lines 43 - 51). Then I employed 2 3x3 2D convolutional layers (cnn.py lines 52-54) each also followed by relu activation and 2d max pooling. Due to the propensity for these models to overfit, I then implimented a dropout layer with a dropout rate of 30% (cnn.py line 55), followed by flattening and 4 fully connected layers before the classification layer (cnn.py lines 56-65).

The model uses the mean squared error to compute loss and the adam optimizer. 20% of the data was split as validation data. Again, due to the model's propensity to overfit (and given the massive amount of data employed), only 3 epochs were run.

#2. Attempts to reduce overfitting in the model

As mentioned, the model employs a dropout layer and few epochs to reduce overfitting.

#3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (cnn.py line 67). However, the steering correction used to account for the left and right offset cameras was adjusted until a comfortable value of .25 was found (cnn.py line 13).

#4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I drove as close as possible to the center of the lane and tried not to veer left or right whatsoever. I also tried to drive at around 13mph to collect a lot of data. I completed 2 laps at this speed.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

#1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the lenet architecture. I thought this model might be appropriate because we are classifying images as lenet does (and was used in the traffic sign classifier project).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and a slightly higher mean squared error on the validation set. This implied that the model might be overfitting. Additionally, the model was unable to handle certain turns in the track, driving the car offroad.

To combat the overfitting, I modified the model so that it had more and larger convolutional layers, and a dropout layer. I also included more fully connected layers to give the model more weights to train. This dramatically reduced the difference in loss between the training and validation data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I included images for all three cameras as well as LR flipped images to increase the amount of data and handle different kinds of turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#2. Final Model Architecture

My model employs a lambda layer to normalize the data (cnn.py line 41), followed by a 2D cropping layer to cut out the sky (cnn.py line 42). Then it has 3 5x5 2D convolutional layers each followed by a relu activation and then max pooling 2D (cnn.py lines 43 - 51). Then I employed 2 3x3 2D convolutional layers (cnn.py lines 52-54) each also followed by relu activation and 2d max pooling. Due to the propensity for these models to overfit, I then implimented a dropout layer with a dropout rate of 30% (cnn.py line 55), followed by flattening and 4 fully connected layers before the classification layer (cnn.py lines 56-65).

The model uses the mean squared error to compute loss and the adam optimizer. 20% of the data was split as validation data. Again, due to the model's propensity to overfit (and given the massive amount of data employed), only 3 epochs were run.

#3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

To augment the data sat, I also flipped images and angles thinking that this would aid the model in learning to turn right, since the track only has a single right turn. For example, here is an image that has then been flipped:


###normal
![alt text][image2]

###flipped
![alt text][image3]

After the collection process, I had over 28,000 of data points. All preprocessing was done in lambda and cropping layers in the model architecture.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by it being producing the smallest difference in loss between the training and validation sets. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The finished model was tested twice, first at 20mph and then at 15mph. While both perform well, they veer left and right alternatively on straight aways a bit. The 15mph test did a much better job in my opinion.