# Kaggle Carvana Image Masking Challenge solution with Keras
This solution was based on [Heng CherKeng's code for PyTorch](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208). I kindly thank him for sharing his work. 128x128, 256x256, 512x512 and 1024x1024 U-nets are implemented. Public LB scores for each U-net are:

| U-net | LB score |
| ----- | -------- |
| 128x128 | 0.990 |
| 256x256 | 0.992 |
| 512x512 | 0.995 |
| 1024x1024 | 0.996 |

---

## Updates

### Update 28.8.2017
* Added loss with weighted boundary (thanks to [lyakaap](https://www.kaggle.com/lyakaap/weighing-boundary-pixels-loss-script-by-keras2))

### Update 15.8.2017
* Added Hue/Saturation/Value augmentation.
* Switched to RMSprop optimizer as default.
* Added multithreaded inference with inference and data loading done on separate threads. This reduced inference time by 40% in my tests. You can run `test_submit_multithreaded.py` to try it.

### Update 10.8.2017
* Added 1024x1024 U-net
* Not using *predict_generator* anymore due to memory constraints with large input

### Update 9.8.2017
* Using *Binary Crossentropy Dice Loss* in place of *Binary Crossentropy*
* Callbacks now use *val_dice_loss* as a metric in place of *val_loss*

---

## Requirements
* Keras 2.0 w/ TF backend
* sklearn
* cv2
* tqdm
* h5py

---

## Usage

### Data
Place '*train*', '*train_masks*' and '*test*' data folders in the '*input*' folder.

Convert training masks to *.png* format. You can do this with: 

` mogrify -format png *.gif` 

in the '*train_masks*' data folder.

### Train
Run `python train.py` to train the model.

### Test and submit
Run `python test_submit.py` to make predictions on test data and generate submission.
