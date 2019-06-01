# SignLanguageRecognition
Sign Language Recognition using a convolutional neural network (CNN) implemented in Keras + TensorFlow + OpenCV.

## Requirements


## Contents
* **images.7z** - compressed images of complete training and validation data.



### Hotkeys
* The ROI on the screen can be moved using the arrows keys.
* Show the binary mask being applied to the ROI by toggling the `m` key.
* Quit the application by pressing the `q` key.


### Taking Data
To collect data, you must first select the target class 

# Feature Input

Images are captured within a ROI from the webcam using OpenCV. To help simplify the inputs that are analyzed by the CNN, a binary mask is applied to highlight the hands edges. The binary mask is defined by grayscaling and blurring the image and then applying thresholding as shown below:

```
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7,7), 3)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

A dataset is collected within the application by holding up 0 to 5 digits at different positions and orientations within the application's ROI. A training set of ~1500 images and validation set of ~600 images for each case is used for training the CNN. 



