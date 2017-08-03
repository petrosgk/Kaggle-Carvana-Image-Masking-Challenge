import pandas as pd
import numpy as np

import cv2
from tqdm import tqdm

from u_net import get_unet_128, get_unet_512

df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

input_size = 128
batch_size = 16

model = get_unet_128()
model.load_weights(filepath='weights/best_weights.hdf5')


def test_generator():
    while True:
        for start in range(0, len(ids_test), batch_size):
            x_batch = []
            end = min(start + batch_size, len(ids_test))
            ids_test_batch = ids_test[start:end]
            for id in ids_test_batch.values:
                img = cv2.imread('input/test/{}.jpg'.format(id))
                img = cv2.resize(img, (input_size, input_size))
                x_batch.append(img)
            x_batch = np.array(x_batch, np.float32) / 255
            yield x_batch


print("Predicting on {} samples".format(len(ids_test)))
preds = model.predict_generator(generator=test_generator(),
                                steps=(len(ids_test) // batch_size))
preds = np.squeeze(preds, axis=3)

orig_width = 1918
orig_height = 1280

threshold = 0.5

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

rles = []
print("Generating masks...")
for pred in tqdm(preds, miniters=1000):
    prob = cv2.resize(pred, (orig_width, orig_height))
    mask = prob > threshold
    rle = run_length_encode(mask)
    rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
