import pandas as pd
import time 
import numpy as np 
from keras import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Reshape, MaxPooling3D, Conv1D, Dropout
import keras 
import os
import sys
import math
import progressbar

import code


def data_generator(chunksize, train_len):
    train_file = '../data/train.csv'
    chunk_folder = '../data/chunks/'
    target_size = int(chunksize)
    tsize = 512
    fsize = 256
    maxval = 20.0
    gridsize = maxval / float(fsize)
    while True:
        chunkx = np.zeros((tsize, 4096))
        chunky = np.zeros((tsize, fsize))
        ii = 0
        for file in os.listdir(chunk_folder):
            df = pd.read_csv(os.path.join(chunk_folder, file))
            data_chunk = df.values[:,0]
            target = np.mean(df.iloc[:,1])
            chunkx[ii, :] = np.zeros((4096, ))
            chunkx[ii, :df.shape[0]] = df.iloc[:min(df.shape[0], 4096),0]
            chunky[ii, :] = np.zeros((fsize, ))
            # ilow = min(math.floor(target/gridsize), 511)
            # ihigh = math.ceil(target/gridsize)
            # chunky[ii, ilow] = 1
            # if ihigh < 256:
            #    chunky[ii, ihigh] = 1.0/(1.0 + abs(ihigh*chunksize-target))
            for k in range(chunky.shape[1]):
                chunky[ii,k] = abs((float(k)*gridsize - target)/gridsize)
            ii += 1
            # yield np.array(data_chunk, dtype=float).reshape(1, 4096), np.array([target, ]).reshape(1, 1)
            # code.interact(banner='', local=locals())
            if ii >= tsize:
                ii = 0
                yield chunkx, chunky
    
def create_network(tlen, get_weights=False):
    filters = 1
    ksize = 512 #1024
    num_cats = 1
    model = Sequential(name='LANL')
    model.add(Dense(4096, input_shape=(tlen,), activation='tanh'))
    # model.add(Reshape((1, tlen, num_cats), input_shape=(tlen, 1)))
    model.add(Dropout(0.1))
    # model.add(Conv2D(kernel_size = (1, num_cats), filters = filters, activation='relu', padding='valid'))
    # model.add(Conv2D(kernel_size = (ksize, ksize), strides=(ksize, ksize), filters = filters, activation='relu', padding='valid'))
    # model.add(Conv2D(kernel_size = (ksize, 1), filters = filters, activation='relu', padding='same'))
    # model.add(Conv2D(kernel_size = (ksize, 1), filters = filters, activation='relu', padding='same'))
    # model.add(Conv2D(kernel_size = (ksize, 1), filters = filters, activation='relu', padding='same'))
    # model.add(Conv2D(kernel_size = (ksize, 1), filters = filters, activation='relu', padding='same'))
    #model.add(Reshape((tlen, num_cats, filters, 1)))
    #model.add(MaxPooling3D(pool_size=(1, 1, filters), strides=(1, 1, filters)))
    # model.add(Flatten())
    # model.add(Dense (tlen, activation='relu'))
    # model.add(Dense (4096, activation='tanh'))
    # model.add(Dropout(0.1))
    model.add(Dense (2048, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense (1024, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense (512, activation='tanh'))
    model.add(Dropout(0.1))    
    model.add(Dense (256, activation='linear'))
    model.compile( loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    # print(model.summary())
    if get_weights:
        loaded = False
        while not loaded:
            try:
                model.load_weights('best_weights.hf5')
                loaded = True
            except:
                print('could not load weights')
                loaded = False 
                time.sleep(1)
    else:
        try:
            model.load_weights('best_weights.hf5')
        except:
            print('could not load weights')
    return model

def get_features(model):
    chunk_folder = '../data/chunks/'
    features = []
    files = os.listdir(chunk_folder)
    bar = progressbar.ProgressBar(len(files))
    bar.start()
    for i, file in enumerate(files):
        bar.update(i)
        feature = dict()
        df = pd.read_csv(os.path.join(chunk_folder, file))
        x = df.iloc[:min(df.shape[0], 4096),0]
        xn = np.zeros((4096, ))
        xn[:len(x)] = x
        # code.interact(banner='', local=locals())
        xhat = model.predict(np.array(xn).reshape(1, -1))
        y = np.mean(df.iloc[:,1])
        # code.interact(banner='', local=locals())
        for i in range(xhat.shape[1]):
            colname = 'col' + str(i)
            feature[colname] = xhat[0, i]
        feature['target'] = y
        features.append(feature)
    feature_df = pd.DataFrame(features)
    feature_df.to_csv('../data/features.csv', index=False)

def predict_segid(segid, model, tlen):
    pcolname = 'time_to_failure'
    segfile = '../data/test/' + segid
    df = pd.read_csv(segfile)
    outputs = []
    for i_start in range(0, df.shape[0]-tlen, int(tlen/2)):
        predict_input = np.array(df.iloc[i_start:(i_start+tlen),0], dtype=float).reshape(1, tlen)
        predict_output = model.predict(predict_input)
        outputs.append(predict_output)
    predict_input = np.array(df.iloc[-tlen:,0], dtype=float).reshape(1, tlen)
    predict_output = model.predict(predict_input)
    outputs.append(predict_output)
    return outputs
    
def compare_to_train(model):
    train_file = '../data/train.csv'
    tlen = 4096
    df_all = pd.read_csv(train_file, chunksize = 4096)
    for df in df_all:
        predict_input = np.array(df.iloc[:,0], dtype=float).reshape(1, tlen)
        p = model.predict(predict_input)
        p_actual = df['time_to_failure'].mean()
        # code.interact(banner='', local=locals())
        print(str(p) + ' - ' + str(p_actual))
    
def read_ttf(segid):
    segfile = '../data/test/' + segid
    df = pd.read_csv(segfile)
    return df.iloc[-1,1]
    
if __name__ == '__main__':
    csize = 4096
    tlen = 4096
    if sys.argv[1] == 'train':
        model = create_network(tlen)
        checkpoint = keras.callbacks.ModelCheckpoint('best_weights.hf5', monitor='mean_absolute_error', verbose=1, save_best_only=False, mode='min')
        model.fit_generator(data_generator(csize, tlen), steps_per_epoch=32, epochs=1000000, callbacks=[checkpoint])
    if sys.argv[1] == 'predict':
        # print("Starting predict")
        files = os.listdir('../data/test/')
        model = create_network(tlen, get_weights=True)
        bar = progressbar.ProgressBar(len(files))
        bar.start()
        idx = 0
        lines = []
        for file in files:
            # print(file)
            bar.update(idx)
            idx += 1
            outputs = predict_segid(file, model, tlen)
            for output in outputs:
                line = { 'segid': file.replace('.csv', '') }
                for i in range(output.shape[1]):
                    colname = 'col' + str(i)
                    line[colname] = output[0, i]
                lines.append(line)
        df = pd.DataFrame(lines)
        df.to_csv('../data/test_features.csv', index=False)
            
    if sys.argv[1] == 'get_features':
        model = create_network(tlen, get_weights=True)
        get_features(model)
    if sys.argv[1] == 'harvest':
        # print("Starting Harvest")
        files = os.listdir('../data/test/')
        bar = progressbar.ProgressBar(len(files))
        preds = []
        bar.start()
        idx = 0
        for file in files:
            # print(file)
            bar.update(idx)
            idx += 1
            ttf = read_ttf(file)
            segid = file.split('.')[0]
            preds.append({'seg_id': segid, 'time_to_failure': ttf})
        # code.interact(banner='', local=locals())
        df = pd.DataFrame(preds)
        df.to_csv('prediction.csv', index=False)
    if sys.argv[1] == 'compare':
        model = create_network(tlen)
        compare_to_train(model)