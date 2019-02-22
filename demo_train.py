import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.util.montage import montage2d as montage
from sklearn.model_selection import train_test_split
from skimage.io import imread
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import time

from utils import *
from loss import *
from data_augument import train_aug
from clr_callback import CyclicLR

from segmentation_models import Unet, FPN
# from segmentation_models.losses import bce_jaccard_loss
# from segmentation_models.metrics import iou_score


BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
MAX_TRAIN_EPOCHS = 300
MAX_TRAIN_STEPS = 1000
AUGMENT_BRIGHTNESS = True
BACKBONE = 'resnet34'

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ship_dir = '../ship_detection/data'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')

masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))
print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0])

# dfWtShipOnly = masks.drop(masks.index[masks.EncodedPixels.apply(lambda x: not isinstance(x, str)).tolist()])
# dfWtShipOnly["rleAndPosition"] = dfWtShipOnly.EncodedPixels.apply(lambda x: ' '.join(x.split(" ")[1::2])
#                                                                             + ' ' + ' '.join(
#     [str(int(hor) % 256) for hor in x.split(" ")[0::2]]) if (isinstance(x, str)) else x)
#
# # List in a new column all the ImageId where the 'rleAndPosition' occurs.
# dfWtShipOnly["allSameRle"] = dfWtShipOnly["rleAndPosition"].apply(
#     lambda x: dfWtShipOnly.ImageId[dfWtShipOnly["rleAndPosition"] == x].tolist())
# dfWtShipOnly.head(10)
# dfWtShipOnly.to_csv('dfWtShipOnly.csv')
# dfWtShipOnly = pd.read_csv('dfWtShipOnly.csv')

# # Group the 'rleAndPosition' by ImageId
# dfWtShipOnlyUnique = dfWtShipOnly.groupby('ImageId')['allSameRle'].apply(lambda x: set(x.sum()))
#
# print(len(dfWtShipOnlyUnique))
# alreadyDropped = []
# dfWtShipOnlyUniqueCopy = dfWtShipOnlyUnique
# for itemKeeped in dfWtShipOnlyUnique.iteritems():
#     if not itemKeeped[0] in alreadyDropped:
#         for itemToCheck in dfWtShipOnlyUnique.iteritems():
#             if itemToCheck[0] in itemKeeped[1] and not itemToCheck[0] in alreadyDropped and itemToCheck[0] != \
#                     itemKeeped[0]:
#                 dfWtShipOnlyUnique = dfWtShipOnlyUnique.drop(itemToCheck[0])
#                 alreadyDropped = alreadyDropped + [itemToCheck[0]]
# print(len(dfWtShipOnlyUnique))
# dfWtShipOnlyUnique.to_csv('dfWtShipOnlyUnique.csv')

# Splitting
# trainDfWtShipOnlyUnique=dfWtShipOnlyUnique.sample(frac=0.9,random_state=768)
# validationDfWtShipOnlyUnique=dfWtShipOnlyUnique.drop(trainDfWtShipOnlyUnique.index)
#
# train_df = masks.loc[[True if ID in trainDfWtShipOnlyUnique.index else False for ID in masks["ImageId"]]]
# valid_df = masks.loc[[True if ID in validationDfWtShipOnlyUnique.index else False for ID in masks["ImageId"]]]

# train_df.to_csv('train_df.csv')
# valid_df.to_csv('valid_df.csv')

train_df = pd.read_csv('train_df.csv')
valid_df = pd.read_csv('valid_df.csv')
# train_df = pd.concat([train_df, valid_df])

print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

aug = train_aug()

def make_image_gen_train(in_df, batch_size=BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            augment = aug(image=c_img, mask=c_mask)
            c_img = augment['image']
            c_mask = augment['mask']
            if c_mask.sum()>50:
                out_rgb += [c_img]
                out_mask += [c_mask]
                if len(out_rgb) >= batch_size:
                    yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                    out_rgb, out_mask = [], []


def make_image_gen_valid(in_df, batch_size=VALID_BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []


train_gen = make_image_gen_train(train_df)
valid_gen = make_image_gen_valid(valid_df)

# define model
seg_model = FPN(BACKBONE, encoder_weights='imagenet',
                classes=1, activation='sigmoid', pyramid_dropout=.2)

weight_path = 'output_201902221308/'+'seg_model_weights.best.hdf5'
seg_model.load_weights(weight_path)

# seg_model.compile(optimizer=Adam(lr=1e-4), loss=bce_jaccard_loss, metrics=[iou_score])
seg_model.compile(optimizer=Adam(), loss=bce_iou_loss, metrics=[iou_coef])
# seg_model.summary()

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
#                                    min_lr=1e-6)
clr = CyclicLR(base_lr=1e-6, max_lr=1e-3, step_size=10)

early = EarlyStopping(monitor="val_iou_coef", mode="max", patience=30)

callbacks_list = [checkpoint, early, clr]

# step_count = train_df.shape[0] // BATCH_SIZE
step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)
loss_history = [seg_model.fit_generator(train_gen,
                                        steps_per_epoch=step_count,
                                        epochs=MAX_TRAIN_EPOCHS,
                                        validation_data=valid_gen,
                                        validation_steps=valid_df.shape[0] // VALID_BATCH_SIZE,
                                        callbacks=callbacks_list,
                                        workers=1)]

seg_model.load_weights(weight_path)
seg_model.save(output_path + 'seg_model.h5')
