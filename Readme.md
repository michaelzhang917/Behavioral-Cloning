# Behavioral-Cloning
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/center.jpg
[image2]: ./examples/recover1.jpg
[image3]: ./examples/recover2.jpg
[image4]: ./examples/recover3.jpg
[image5]: ./examples/recover4.jpg "recover"
[image6]: ./examples/normal.jpg
[image7]: ./examples/brightness.jpg
[image8]: ./examples/translate.jpg
[image9]: ./examples/shadow.jpg
[image10]: ./examples/flipped.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

I implement the model based on paper "End to End Learning for Self-Driving Cars" using Keras. The following is the 
details of the model.

* Layer 1: The data is normalized to [-0.5, 0.5] by using a Keras lambda layer (code line 26)
* Layer 2: A batch normalization is used to further normalize the data (code line 27)
* Layer 3: A convolution layer with 3 3x3 filter size,  RELU activation (code line 28)
* Layer 4: A Maxpooling layer with 2x2 size (code line 29)
* Layer 5: A convolution layer with 9 3x3 filter size,  RELU activation (code line 31)
* Layer 6: A Maxpooling layer with 2x2 size (code line 32)
* Layer 7: A convolution layer with 18 3x3 filter size,  RELU activation (code line 34)
* Layer 8: A Maxpooling layer with 2x2 size (code line 35)
* Layer 9: A convolution layer with 32 3x3 filter size,  RELU activation (code line 37)
* Layer 10: A Maxpooling layer with 2x2 size (code line 38)
* Layer 11: A fully connected layer with 80 output and RELU activation (code line 41)
* Layer 12: A fully connected layer with 15 output and RELU activation (code line 43)
* Layer 13: A output layer using linear activation (code line 46)

I also tried to use VGG16 based model but I gave up because it was too large when I deployed the model on my laptop to 
run the simulator.

####2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 45). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 245-247). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 206).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to correctly predict a steering angle for driving the car in the provided simulator.

My first step was to use a convolution neural network model similar to the VGG16 model. I thought this model might be appropriate because the VGG16 model can extract valuable features from the image.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and a low mean squared error on the validation set. However, when I deploy this model to my laptop, the car can not be driven properly becaue my laptop is not fast enough to do the CNN inference and run the simulator in the same time. 

This means my CNN has too many variables.

Then I read the paper "End to End Learning for Self-Driving Cars"  and build a much smaller model. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and can not recover. To improve the driving behavior in these cases, I have driven more circles to generate more data points. Especially, at the failed spots, I drove from the side to the center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

See the video link here https://www.youtube.com/watch?v=2K8WQtBizr0

The preformance actually was downgraded in the video because screen recording affected the preformance.


####2. Final Model Architecture

The final model architecture has been described in Section 1.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the middle. These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

I also collected the sample images from https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

To augment the data set, I also randomly adjust the brightness of the images, translate the images and its angle, add random shadow to the imagse and flip the images and angles and hope this would increase the robustness of the model. For example, here is the birghtness of an image that has then been adjusted :

![alt text][image6]
![alt text][image7]

Here is the image that has been translated (for each pixel translated pixel, the steering angle is corrected by 0.004.

![alt text][image6]
![alt text][image8]

Here is the image that has been added random shadow

![alt text][image6]
![alt text][image9]

Here is the image that has been flipped

![alt text][image6]
![alt text][image10]

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
