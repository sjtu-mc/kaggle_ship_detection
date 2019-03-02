# Kaggle Airbus Ship Detection Challenge
This is an implementation of the Kaggle competition  [Airbus Ship Detection Challenge | Kaggle](https://www.kaggle.com/c/airbus-ship-detection) . 

The project include:
 * training process with Unet or FPN, which is implemented by [GitHub - qubvel/segmentation_models: Segmentation models with pretrained backbones. Keras.](https://github.com/qubvel/segmentation_models)
 * data augmentations implemented by [GitHub - albu/albumentations: fast image augmentation library and easy to use wrapper around other libraries](https://github.com/albu/albumentations)
 * cycle learning rate (CLR) implemented by [GitHub - bckenstler/CLR](https://github.com/bckenstler/CLR)
 * test procedure with TTA and the linear search of the best threshold
## Recent Update
* 2019.2.22 first commit
* 2019.3.2 modify the data generate, change the `demo_test` filetype from `.ipynb` to `.py`, change the training procedure 
## Dependencies
* python==3.6.7
* Keras==2.1.2
* tensorflow==1.4.1
## Steps
### Training
Run `./demo_train.py`, then you will get the segmentation model in the `“output_” + current time` fold. 
### Test
Run  `./demo_test.py` with the Jupyter notebook, finally you will get the submission file `submission.csv` in the `“output_” + current time` fold.
