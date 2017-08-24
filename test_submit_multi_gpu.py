import cv2
import numpy as np
import pandas as pd
import threading
import queue
import tensorflow as tf
from tensorflow.python.client import device_lib
from tqdm import tqdm

import params

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model_factory = params.model_factory

gpus = [x.name for x in device_lib.list_local_devices() if x.name[:4] == '/gpu']

df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
rles = []
q_size = 10

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

def create_model(gpu):
    with tf.device(gpu):
        model = model_factory()
    model.load_weights(filepath='weights/best_weights.hdf5')
    return model

def data_loader(q, ):
    for start in tqdm(range(0, len(ids_test), batch_size)):
        x_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch.values:
            img = cv2.imread('input/test/{}.jpg'.format(id))
            if input_size is not None:
                img = cv2.resize(img, (input_size, input_size))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        q.put((ids_test_batch, x_batch))
    for g in gpus:
        q.put((None, None))

def predictor(q, gpu):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    with sess.as_default():
        model = create_model(gpu)
        while True:
            ids, x_batch = q.get()
            if ids is None:
                break
            preds = model.predict_on_batch(x_batch)
            preds = np.squeeze(preds, axis=3)
            for i,pred in enumerate(preds):
                if input_size is not None:
                    prob = cv2.resize(pred, (orig_width, orig_height))
                else:
                    prob = pred
                mask = prob > threshold
                rle = run_length_encode(mask)
                id = ids.iloc[i]
                rles.append((id, rle))



print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
q = queue.Queue(maxsize=q_size)
threads = []
threads.append(threading.Thread(target=data_loader, name='DataLoader', args=(q,)))
threads[0].start()
for gpu in gpus:
    print("Starting predictor at device " + gpu)

    t = threading.Thread(target=predictor, name='Predictor', args=(q, gpu))
    threads.append(t)
    t.start()

# Wait for all threads to finish
for t in threads:
    t.join()

print("Generating submission file...")
df = pd.DataFrame(rles, columns=['img', 'rle_mask'])
df['img'] += '.jpg'
df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
