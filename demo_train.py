import os
import time
import warnings

warnings.filterwarnings('ignore')

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.util.montage import montage2d as montage
from sklearn.model_selection import train_test_split

from clr_callback import CyclicLR
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import SGD, Adam
from keras.utils import multi_gpu_model

from loss import *
from segmentation_models import FPN
from utils import *
from data_generate import *

BATCH_SIZE = 24
VALID_BATCH_SIZE = 8
MAX_TRAIN_EPOCHS = 300
MAX_TRAIN_STEPS = 1000
AUGMENT_BRIGHTNESS = True
BACKBONE = 'resnet34'

ship_dir = '../ship_detection/data'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')

masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))
print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0])

masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
masks.drop(['ships'], axis=1, inplace=True)

dfWtShipOnly = masks.drop(masks.index[masks.EncodedPixels.apply(lambda x: not isinstance(x, str)).tolist()])
dfWtShipOnly["rleAndPosition"] = dfWtShipOnly.EncodedPixels.apply(lambda x: ' '.join(x.split(" ")[1::2])
                                                                            + ' ' + ' '.join(
    [str(int(hor) % 256) for hor in x.split(" ")[0::2]]) if (isinstance(x, str)) else x)

# List in a new column all the ImageId where the 'rleAndPosition' occurs.
dfWtShipOnly["allSameRle"] = dfWtShipOnly["rleAndPosition"].apply(
    lambda x: dfWtShipOnly.ImageId[dfWtShipOnly["rleAndPosition"] == x].tolist())

# Group the 'rleAndPosition' by ImageId
dfWtShipOnlyUnique = dfWtShipOnly.groupby('ImageId')['allSameRle'].apply(lambda x: set(x.sum()))
alreadyDropped = []
dfWtShipOnlyUniqueCopy = dfWtShipOnlyUnique
for itemKeeped in dfWtShipOnlyUnique.iteritems():
    if not itemKeeped[0] in alreadyDropped:
        for itemToCheck in dfWtShipOnlyUnique.iteritems():
            if itemToCheck[0] in itemKeeped[1] and not itemToCheck[0] in alreadyDropped and itemToCheck[0] != \
                    itemKeeped[0]:
                dfWtShipOnlyUnique = dfWtShipOnlyUnique.drop(itemToCheck[0])
                alreadyDropped = alreadyDropped + [itemToCheck[0]]
dfWtShipOnlyUnique.to_csv('dfWtShipOnlyUnique.csv')

dfWtShip = pd.read_csv('dfWtShipOnlyUnique.csv')
dfWtShip.columns = ['ImageId','DuplicateId']
dfWtShip = pd.DataFrame(dfWtShip['ImageId'])
print(len(dfWtShip))

dfWtNoShip = unique_img_ids[unique_img_ids['has_ship']==0]
dfWtNoShip = pd.DataFrame(dfWtNoShip['ImageId'])
dfWtNoShip = dfWtNoShip.sample(n=int(len(dfWtShip)/0.3),random_state=768)
print(len(dfWtNoShip))

df_all = pd.concat([dfWtNoShip,dfWtShip])

# Splitting
trainDf=dfWtShip.sample(frac=0.9,random_state=768)
validationDf=dfWtShip.drop(trainDf.index)

train_df = masks[masks.ImageId.isin(trainDf.ImageId)]
valid_df = masks[masks.ImageId.isin(validationDf.ImageId)]
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

train_df.to_csv('train_df.csv')
valid_df.to_csv('valid_df.csv')

train_sequence = SequenceTrainData(train_df, BATCH_SIZE, train_image_dir)
valid_sequence = SequenceValidData(valid_df, VALID_BATCH_SIZE, train_image_dir)

# define model
seg_model = FPN(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid', pyramid_dropout=.2)

optimizer = Adam()
lr_metric = get_lr_metric(optimizer)
seg_model.compile(optimizer=optimizer, loss=bce_iou_loss, metrics=[iou_coef,lr_metric])

current_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
output_path = './output_' + current_time + '/'
print(output_path)
folder = os.path.exists(output_path)
if not folder:
    os.makedirs(output_path)

weight_path = output_path + 'seg_model_weights.best.hdf5'

checkpoint = ModelCheckpoint(weight_path, monitor='val_iou_coef', verbose=1, save_best_only=True, mode='max',
                             save_weights_only=True)

# reduceLROnPlat = ReduceLROnPlateau(monitor='val_iou_coef', factor=0.5, patience=3, verbose=1, mode='max',
                                #    min_lr=1e-6)
clr = CyclicLR(base_lr=1e-6, max_lr=1e-3, step_size=500)

early = EarlyStopping(monitor="val_iou_coef", mode="max", patience=30)

log_vision = TensorBoard(log_dir=output_path)

callbacks_list = [checkpoint, early, clr, log_vision]

loss_history = [seg_model.fit_generator(train_sequence,
                                        epochs=MAX_TRAIN_EPOCHS,
                                        validation_data=valid_sequence,
                                        callbacks=callbacks_list,
                                        workers=4,
                                        use_multiprocessing=True)]

seg_model.load_weights(weight_path)
seg_model.save(output_path + 'seg_model.h5')
