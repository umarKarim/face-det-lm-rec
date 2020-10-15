# Face Detection, Landmark Detection and Face Recognition in TensorFlow
## Introduction
This repository contains inference code for face detection, landmark detection and face recognition in tensorflow. The repository allows basic understanding of the above tasks using deep neural networks. Also, this repo allows easy testing for Python users of the face models provided by Google. These include

- BlazeFace
- Face Mesh
- Facenet

The models are *.tflite* so they can be used for smartphone applications as well.


## Requirements
- TensorFlow 
- Numpy 
- OpenCV

Although I have installed tensorflow-gpu, the code runs on cpu. In other words, tensorflow-gpu may not be necessary.

## How to use
The parameters for the code are given in the *options.py* file. Change them to play around. Basic usage in bash is as follows. 

```
python3 face_processor.py --test_im_name <image_path>
```

The face crop and face crop with landmarks will be stored in the *out_dir*. Console output will provide the ID of the face from the *face_database*.

## Models 
For face detection, I use the Blaze Face model from Google mediapipe. For face landmarks detection, I use the Face Mesh model from Google mediapipe. For face recognition, I extract features using the FaceNet model from [link](https://github.com/estebanuri/face_recognition/tree/master/android/models). Parts of face detection code are from [link](https://github.com/hollance/BlazeFace-PyTorch).

## Some Results
### Good result
![Input](https://github.com/umarKarim/face-det-lm-rec/tree/master/face_ims/test_ims/momoa_2.jpg)
![face crop](https://github.com/umarKarim/face-det-lm-rec/tree/master/outputs/face_crop_ok.png)
![face landmarks](https://github.com/umarKarim/face-det-lm-rec/tree/master/outputs/landmarks_ok.png)
### Bad result
![Input](https://github.com/umarKarim/face-det-lm-rec/tree/master/face_ims/test_ims/streep_2.jpg)
![face crop](https://github.com/umarKarim/face-det-lm-rec/tree/master/outputs/face_crop_fail.png)
![face landmarks](https://github.com/umarKarim/face-det-lm-rec/tree/master/outputs/landmarks_fail.png)

## Discussion 
Face mesh sometimes fails if the face is too tilted in any direction. I think the problem is common to landmark detection schemes.
Also, the code is not for videos. If you want to apply it to videos then apply a temporal smoothing filter to the landmarks. It will reduce the jitter in landmarks across frames. 
Are the bad results because of my code or are they because of the models? Please let me know.







