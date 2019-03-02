BATCH_SIZE = 8
output_path = 'output_201903011841/'
weight_save_path = "output_201903011841/"
BACKBONE = 'resnet34'

import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from segmentation_models import FPN
from keras.utils import multi_gpu_model
from tqdm import tqdm
from data_generate import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from utils import *

ship_dir = '../ship_detection/data'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')

aug_valid = valid_aug()
def make_image_gen(in_df, batch_size = BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            augment = aug_valid(image=c_img, mask=c_mask)
            c_img = augment['image']
            c_mask = augment['mask']
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0), np.stack(out_mask, 0)

valid_df = pd.read_csv('valid_df.csv')
valid_gen = make_image_gen(valid_df, 500)
valid_x, valid_y = next(valid_gen)
print(valid_x.shape, valid_y.shape)

seg_model = FPN(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid', pyramid_dropout=.2)

class TTA_ModelWrapper():
    """A simple TTA wrapper for keras computer vision models.
    Args:
        model (keras model): A fitted keras model with a predict method.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X, verbose=1):
        """Wraps the predict method of the provided model.
        Augments the testdata with horizontal and vertical flips and
        averages the results.
        Args:
            X (numpy array of dim 4): The data to get predictions for.
        """
        p0 = self.model.predict(X, batch_size=BATCH_SIZE, verbose=verbose)
        p1 = self.model.predict(np.fliplr(X), batch_size=BATCH_SIZE, verbose=verbose)
        p2 = self.model.predict(np.flipud(X), batch_size=BATCH_SIZE, verbose=verbose)
        p3 = self.model.predict(np.transpose(X, (0, 2, 1, 3)), batch_size=BATCH_SIZE, verbose=verbose)
        p4 = self.model.predict(np.rot90(X, 1, (1, 2)), batch_size=BATCH_SIZE, verbose=verbose)
        p5 = self.model.predict(np.rot90(X, 2, (1, 2)), batch_size=BATCH_SIZE, verbose=verbose)
        p6 = self.model.predict(np.rot90(X, 3, (1, 2)), batch_size=BATCH_SIZE, verbose=verbose)
        p7 = self.model.predict(np.rot90(np.transpose(X, (0, 2, 1, 3)), 2), batch_size=BATCH_SIZE, verbose=verbose)
        #         print(p7.shape)
        #         print(np.rot90(np.transpose(p7),2).shape)
        p = (p0 +
             (np.fliplr(p1)) +
             (np.flipud(p2)) +
             (np.transpose(p3, (0, 2, 1, 3))) +
             (np.rot90(p4, 3, (1, 2))) +
             (np.rot90(p5, 2, (1, 2))) +
             (np.rot90(p6, 1, (1, 2))) +
             (np.rot90(np.transpose(p7, (0, 2, 1, 3)), 2))
             ) / 8

        return p

    def _expand(self, x):
        return np.expand_dims(x, axis=0)

tta_model = TTA_ModelWrapper(seg_model)
pred_y = tta_model.predict(valid_x)

print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())

thresholds = np.linspace(0, 1, 10)
ious = np.array([iou_metric_batch(valid_y, np.int32(pred_y > threshold)) for threshold in tqdm(thresholds)])

threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
print('The best IoU is', iou_best, 'the best threshold is', threshold_best)

def predict(img, path=test_image_dir):
    c_img = imread(os.path.join(path, img))
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = tta_model.predict(c_img,0)
    
    cur_seg = np.array(np.round(cur_seg[0,:,:,:] > threshold_best), dtype=np.float32)
#     cur_seg = binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))
    return cur_seg, c_img

test_paths = np.array(os.listdir(test_image_dir))
print(len(test_paths), 'test images found')

def pred_encode(img, **kwargs):
    cur_seg, _ = predict(img)
    cur_rles = multi_rle_encode(cur_seg, **kwargs)
    return [[img, rle] for rle in cur_rles if rle is not None]

out_pred_rows = []
for c_img_name in tqdm(test_paths):   #boat_df.id[is_boat]
    out_pred_rows += pred_encode(c_img_name)

sub = pd.DataFrame(out_pred_rows)
sub.columns = ['ImageId', 'EncodedPixels']
sub = sub[sub.EncodedPixels.notnull()]
sub.head()

sub1 = pd.read_csv('../ship_detection/data/sample_submission_v2.csv')
sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])
sub1['EncodedPixels'] = None
print(len(sub1), len(sub))

sub = pd.concat([sub, sub1])
print(len(sub))

sub.to_csv(output_path+'submission'+'.csv', index=False)
sub.head()
