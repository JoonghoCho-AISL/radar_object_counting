import os
from tqdm import tqdm
import h5py
import numpy as np
from multiprocessing import Pool

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def read_data(data_path):
    f = h5py.File(data_path, 'r')
    data = f['session']['group_0']['entry_0']['result']['frame']
    temp_i = list()
    for i in tqdm(range(len(data))):
        temp_j = list()
        for j in range(len(data[i])):
            temp_k = list()
            for k in range(len(data[i][j])):
                temp_k.append(np.abs(complex(data[i][j][k][0], data[i][j][k][1])))
            temp_j.append(np.array(temp_k))
        temp_i.append(np.array(temp_j))
    return np.array(temp_i)


def dataSaver(file):
    global save_path
    save_folder = os.path.split(os.path.dirname(file))[1]
    save_file_path = os.path.join(save_path, save_folder, os.path.split(file)[1])
    createDirectory(os.path.dirname(save_file_path))
    print(os.path.dirname(save_file_path))
    data = read_data(file)
    np.save(save_file_path, data)


if __name__=='__main__':
    folder_path = './Data/'
    save_path = './Data_npy'
    file_list = list()
    for folder in os.listdir(folder_path):
        d = os.path.join(folder_path, folder)
        if os.path.isdir(d):
            for file in os.listdir(d):
                if file != '.DS_Store':
                    file_list.append(os.path.join(d, file))
    # dataSaver(file_list[0])                
    with Pool(64) as p:
        p.map(dataSaver, file_list)