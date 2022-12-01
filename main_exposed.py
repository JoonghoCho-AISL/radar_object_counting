import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import os
from acconeer.exptool import a121
from sklearn.model_selection import train_test_split
import argparse

import pickle as pkl

from sklearn.decomposition import PCA

import models

from preprocess import processing

random_seed = 33

label2idx_Dict = {
                '0' : 0,
                '1' : 1,
                '2' : 2,
                '3' : 3,
                '4' : 4,
                '5' : 5,
                '6' : 6,
            }

idx2label_Dict = {
    0 : '0',
    1 : '1',
    2 : '2',
    3 : '3',
    4 : '4',
    5 : '5',
    6 : '6'
}
def readData(data_path):
    data_dict = {
    '0' : dict(),
    '1' : dict(),
    '2' : dict(),
    '3' : dict(),
    '4' : dict(),
    '5' : dict(),
    '6' : dict(),
    }
    for i in os.listdir(data_path):
        d = os.path.join(data_path, i)
        if os.path.isdir(d):
            for file in os.listdir(d):
                if i == '0' or i == '6':
                    length = 1800
                else : length = 300
                if file != '.DS_Store':
                    data_dict[i][file] = np.mean(np.load(os.path.join(d, file))[:length], axis = 1)
    return data_dict
    
def createDatadict(Dict : dict) -> dict :
    Data = dict()
    for i in Dict:
        temp = None
        for j in Dict[i]:
            if type(temp) == type(None):
                temp = Dict[i][j]
            else:
                temp = np.concatenate((temp, Dict[i][j]), axis = 0)
        Data[i] = temp
    return Data

def concat(arr : list):
        temp = arr[0]
        for i in range(1, len(arr)):
            temp = np.concatenate((temp, arr[i]), axis = 0)
        return temp

def make_ds(data : dict):
    temp_arr = list()
    # temp_dict = dict()
    for i in data:
        # for i in data:
        temp_arr.append(add_label(data[i], i))
    temp_data = concat(temp_arr)
    Y = to_categorical(temp_data[:,-1], num_classes = len(label2idx_Dict))
    X_train, X_test, Y_train, Y_test = train_test_split(temp_data[:,:-1], Y, random_state = random_seed)
    return X_train, X_test, Y_train, Y_test

def add_label(arr, label):
        # label = list()
        # for i in idx2label_Dict:
        #     label.append(idx2label_Dict[i])
        label_list = [label2idx_Dict[label] for j in range(len(arr))]
        label_list = np.array(label_list)
        label_list = np.reshape(label_list, (len(arr), 1))
        # print(label_list.shape)
        labeled_arr = np.concatenate((arr, label_list), axis = 1)
        return labeled_arr

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def saveplot(history, plot_name):
    plot_name =  str(plot_name) + '.png'
    plt.subplot(4,1,1)
    plt.ylim(0,1)
    max = np.argmax(history['val_accuracy'])
    plt.plot(history['val_accuracy'])
    plt.plot(max, history['val_accuracy'][max], 'o', ms = 10, label=round(history['val_accuracy'][max],4))
    plt.title('val_accuracy')
    plt.legend(loc = 'center right',fontsize=15)

    plt.subplot(4,1,2)
    plt.ylim(0,1)
    min = np.argmin(history['val_loss'])
    plt.plot(history['val_loss'])
    plt.plot(min, history['val_loss'][max], 'o', ms = 10, label=round(history['val_loss'][max],4))
    plt.title('val_loss')
    plt.legend(loc = 'center right',fontsize=15)

    plt.subplot(4,1,3)
    plt.ylim(0,1)
    max = np.argmax(history['accuracy'])
    plt.plot(history['accuracy'])
    plt.plot(max, history['accuracy'][max], 'o', ms = 10, label=round(history['accuracy'][max],4))
    plt.title('accuracy')
    plt.legend(loc = 'center right',fontsize=15)

    plt.subplot(4,1,4)
    plt.ylim(0,1)
    min = np.argmin(history['loss'])
    plt.plot(history['loss'])
    plt.plot(min, history['loss'][max], 'o', ms = 10, label=round(history['loss'][max],4))
    plt.title('loss')
    plt.legend(fontsize=15)
    plt.legend(loc = 'center right',fontsize=15)
    createDirectory(os.path.dirname(plot_name))
    plt.savefig(plot_name, dpi = 300)
    
# def saveplot(history, plot_name):
#     plot_name =  str(plot_name) + '.png'
#     # plt.subplot(2,1,1)
#     plt.figure(figsize=(12,5))
#     plt.ylim(0,1)
#     max = np.argmax(history['val_accuracy'])
#     plt.plot(history['val_accuracy'])
#     plt.plot(max, history['val_accuracy'][max], 'o', color = 'k', ms = 10, label=('max acc : ' + str(round(history['val_accuracy'][max],4))))
#     plt.title('test accuracy')
#     plt.legend(loc = 'center right',fontsize=15)
#     plt.savefig(plot_name, dpi = 300)

def make_log(y, y_pred):
    Y = np.argmax(y, axis = 1)
    Y_pred = np.argmax(y_pred, axis = 1)
    log_y = [idx2label_Dict[Y[i]] for i in range(len(Y))]
    log_y_pred = [idx2label_Dict[Y_pred[i]] for i in range(len(Y_pred))]
    arr = np.array([log_y, log_y_pred])
    df = pd.DataFrame(arr)
    df.index = ['Ground Truth', 'Prediction']
    df.to_csv('./report_log.csv')

def train(
    model, X, Y,
    test_X, test_Y, 
    # callback,
    Epoch = 50,
    learning_rate = 1e-3,
    cp_path = None,
    ):
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = './model/' + cp_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only = True,
        save_weigths_only = False,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer = optimizer , loss = loss, 
                metrics = ['accuracy', 'categorical_crossentropy'])
    print('X shape : ', X.shape)
    model.build(input_shape = (1, X.shape[1]))          
    model.summary()
    history = model.fit(X, Y,  epochs = Epoch,
                validation_data = (test_X, test_Y),
                callbacks = [callback]
                )
    return history

def main():
    
    parser = argparse.ArgumentParser(description = 'selt data and gpu')
    parser.add_argument('-g', '--gpu', action = 'store', default = '3')
    parser.add_argument('-d', '--data', action = 'store')
    parser.add_argument('-m', '--model', action = 'store', default = 'base')
    args = parser.parse_args()

    path = '/data_disk/home/joongho/ObjectCouning_Radar/Data_npy'
    gpu = int(args.gpu)
    data = args.data
    # Pca = args.pca
    sel_model = args.model
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
        except RuntimeError as e:
            # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
            print(e)

    npy_data = readData(path)
    Data = createDatadict(npy_data)
    DATA = make_ds(Data)
    
    X_train, Y_train, X_test, Y_test = DATA[0], DATA[2], DATA[1], DATA[3]

    if sel_model == 'base':
        title = './result/basemodel'
        model = models.basemodel()
        history = train(model, X_train, Y_train, X_test, Y_test, Epoch = 50, cp_path=title)
        saveplot(history.history, title)
    elif sel_model == 'ann':
        title = './result/ann'
        model = models.ann()
        history = train(model, X_train, Y_train, X_test, Y_test, Epoch = 50, cp_path=title)
        saveplot(history.history, title)
    elif sel_model == 'cnn':
        title = './result/cnn'
        model = models.cnn()
        history = train(model, X_train, Y_train, X_test, Y_test, Epoch = 50, cp_path=title)
        saveplot(history.history, title)
if __name__ == '__main__':
    main()