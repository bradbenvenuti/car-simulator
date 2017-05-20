# Behavioral Cloning

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

![alt text][image1]

My Model is based on the NVIDIA architecture discussed in the lesssons.

The model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 48 and strides of 2x2. Each convolution has a RELU activation.

Next, the model consists of more convolution layers with 3x3 filter sizes and depths of 64 and strides of 1x1. Each convolution has a RELU activation.

Next, the model has a flatten layer and 4 dense layers that output 100, 50, 10, 1.

The model includes RELU layers to introduce nonlinearity, and the data is normalized (code line 79) and cropped (code line 82) in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The training data was created from two different tracks, driven clockwise and counterclockwise, with recovery laps (driving from the edge to center of the track), with 3 camera angles, and each image flipped horizontally. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. It easily drove around track 1. On track two it struggled around a very sharp turn, hitting the edge.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 98).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it was designed for self-driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that it used dropout.

Then I added a lot more training data, including driving the track counterclockwise, adding recovery laps and driving the second track. I also used 3 camera images with adjusted steering measurements and duplicated and flipped every image.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, but on track two the car drover perfectly. To improve the driving behavior in track 1, I trained the model again making sure to shuffle the data first. It seemed that the model was favoring the second track and shuffling the data resolved that.

At the end of the process, the vehicle is able to drive autonomously around track 1 without leaving the road. It was also able to make it a good portion of track 2.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer (type) | Output Shape  |
| ------------ | ------------- |
| <keras.layers.core.Lambda> | (None, 160, 320, 3) |
| <keras.layers.convolutional.Cropping2D> | (None, 70, 320, 3) |
| <keras.layers.convolutional.Convolution2D> | (None, 33, 158, 24) |
| <keras.layers.convolutional.Convolution2D> | (None, 15, 77, 36) |
| <keras.layers.convolutional.Convolution2D> | (None, 6, 37, 48) |
| <keras.layers.convolutional.Convolution2D> | (None, 4, 35, 64) |
| <keras.layers.convolutional.Convolution2D> | (None, 2, 33, 64) |
| <keras.layers.core.Flatten> | (None, 4224) |
| <keras.layers.core.Dense> | (None, 100) |
| <keras.layers.core.Dense> | (None, 50) |
| <keras.layers.core.Dense> | (None, 10) |
| <keras.layers.core.Dense> | (None, 1) |

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
