# Headpose_Estimation

Implementation of the paper in headpose estimation(https://arxiv.org/abs/1812.00739) on the BIWI kinect headpose data(https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#db).

### Nose, eyes and ears: Head pose estimation by locating facial keypoints

## Tech/FrameWork Used

Python 3.7.3

Keras with tensorflow as Backend

Openpose(https://github.com/CMU-Perceptual-Computing-Lab/openpose)

Dlib


## Paper Objective

The paper is based on the hypothesis of learning a model for head pose estimation which relies only on five facial keypoint locations and abstracts out the dependency on appearance of the subject. A  CNN-based  framework  which  uses  the  probability  distribution  of  keypoint  locations  in  the  form  of heatmap images, as input to regress the head pose.

## Implementation Details
The given images in the BIWI dataset has a wide perspective and hence it is important to get the face crops which is done using Dlib's convolutional face detector.</br>
  
Original Image - 

![Original](https://github.com/Arnav0400/Headpose_Estimation/blob/master/image.png "Original")
      
Cropped Image-

<img src="https://github.com/Arnav0400/Headpose_Estimation/blob/master/crop.png"
     width="128" height="128" class="center" />
     
Facial Heatmaps are extracted from the cropped image using openpose 

![Right Ear](https://github.com/Arnav0400/Headpose_Estimation/blob/master/Rear.png "Right Ear") ![Left Ear](https://github.com/Arnav0400/Headpose_Estimation/blob/master/Lear.png "Left Ear") ![Right Eye](https://github.com/Arnav0400/Headpose_Estimation/blob/master/Reye.png "Right Eye") ![Left Eye](https://github.com/Arnav0400/Headpose_Estimation/blob/master/Leye.png "Left Eye") ![Nose](https://github.com/Arnav0400/Headpose_Estimation/blob/master/nose.png "Nose") ![Background](https://github.com/Arnav0400/Headpose_Estimation/blob/master/bkg.png "Background")<img src="https://github.com/Arnav0400/Headpose_Estimation/blob/master/crop.png"
     alt="Original Image"
     width="96" height="96" class="center" />

These Five heatmaps are stacked together and passed into the CNN for regression  

The Model-  
```ruby
inpu = Input(shape = (96,96,5))
x = Conv2D(50, (5, 5), activation=None, padding='valid')(inpu)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(50, (5, 5), activation=None, padding='valid')(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(150, (5, 5), activation=None, padding='valid')(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(300, activation='tanh')(x)
x = Dropout(0.3)(x)
x = Dense(300, activation='tanh')(x)
x = Dropout(0.3)(x)
pred = Dense(3,activation='tanh')(x)
model = Model(inputs = inpu,outputs = pred)
```




# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
