import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class basemodel(Model):
    def __init__(self):
        super(basemodel, self).__init__()
        self.nl1 = layers.Normalization(axis = -1)
        self.fc1 = layers.Dense(units = 10, activation = 'relu')
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(units = 5, activation = 'relu')
        self.out = layers.Dense(units = 7, activation = 'softmax')
    
    def call(self, x, training = False, mask = None):
        out = self.nl1(x)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.out(out)
        return out

class ann(Model):
    def __init__(self):
        super(ann, self).__init__()
        self.fc1 = layers.Dense(units = 10, activation = 'relu')
        self.fc2 = layers.Dense(units = 5, activation = 'relu')
        self.dr = layers.Dropout(0.2)
        self.out = layers.Dense(units = 7, activation = 'softmax')
    def call(self, x , training = False, mask = None):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.out(out)
        return out

def create_model():
    model = tf.keras.models.Sequential([
        layers.Normalization(axis = -1),
        layers.Dense(units = 10, activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dense(units=5, activation = 'relu'),
        layers.Dense(units = 1, activation = 'relu')
        ])
    return model

class cnn(Model):
    def __init__(self):
        super(cnn, self).__init__()
        # self.bn1 = layers.BatchNormalization()
        self.cnn1 = layers.Conv1D(64, (3), activation = 'relu', padding = 'same')
        self.cnn2 = layers.Conv1D(64, (3), activation = 'relu', padding = 'same')
        self.pool1 = layers.MaxPool1D(2)
        self.cnn3 = layers.Conv1D(128, (3), activation = 'relu', padding = 'same')
        self.bn2 = layers.BatchNormalization()
        self.cnn4 = layers.Conv1D(128, (3), activation = 'relu', padding = 'same')
        self.bn3 = layers.BatchNormalization()
        self.pool2 = layers.MaxPool1D(2)
        self.fl = layers.Flatten()
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(128)
        self.fc3 = layers.Dense(7)
    
    def call(self, x, training = False, mark = None):
        x = tf.expand_dims(x, axis = -1)
        # out = self.bn1(x)
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = self.pool1(out)
        out = self.cnn3(out)
        out = self.bn2(out)
        out = self.cnn4(out)
        out = self.bn3(out)
        out = self.pool2(out)
        out = self.fl(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
