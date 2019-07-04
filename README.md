# Headpose_Estimation

Implementation of the state of the art paper in headpose estimation(https://arxiv.org/abs/1812.00739) on the BIWI kinect headpose data(https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#db).

### Nose, eyes and ears: Head pose estimation by locating facial keypoints

## Tech/FrameWork Used

Python 3.7.3

Keras with tensorflow as Backend

Openpose(https://github.com/CMU-Perceptual-Computing-Lab/openpose)

Dlib


## Paper Objective

The paper is based on the hypothesis of learning a model for head pose estimation which relies only on five facial keypoint locations and abstracts out the dependency on appearance of the subject. A  CNN-based  framework  which  uses  the  probability  distribution  of  keypoint  locations  in  the  form  of heatmap images, as input to regress the head pose.

## Implementation Details


# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
