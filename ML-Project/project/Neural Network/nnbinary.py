import numpy as np
from scipy.special import expit as sgm
from matplotlib import pyplot as plt
import cv2
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def import_data():
    global dataset
    global dataset1
    url = "../Dataset2/NEW_myX"
    url1 = "../Dataset2/NEW_myX_test"
    # names = ['Feature 1', 'Feature 2', 'class']
    # dataset = pandas.read_csv(url, sep=" ")
    dataset = np.loadtxt(url, dtype=np.uint8)
    dataset1 = np.loadtxt(url1, dtype=np.uint8)
    # print(dataset)




if __name__ == '__main__':
    # input_data, output_data, size = return_data('./steering')
    import_data()
    np.random.shuffle(dataset)
    np.random.shuffle(dataset1)
    X = dataset[:, 0:200]
    input_data = np.asarray(X)
    y = dataset[:, 200]
    X1 = dataset1[0:1000, 0:200]
    y1 = dataset1[0:1000, 200]
    y = y[:, None]
    y1 = y1[:, None]
    output_data = np.asarray(y)
    output_data = output_data.reshape((len(output_data), 1))
    input_data1 = np.asarray(X1)
    output_data1 = np.asarray(y1)
    output_data1 = output_data1.reshape((len(output_data1), 1))
    #Initializing Neural Network
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'softmax', input_dim = 200))
    # Adding the second hidden layer
    # classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'sigmoid'))
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling Neural Network
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Fitting our model 
    history = classifier.fit(X, y, validation_split=0.33, batch_size = 64,nb_epoch = 50)
    # Predicting the Test set results
    y_pred = classifier.predict(X1)
    y_pred = (y_pred > 0.4)
    # print(y1.shape)
    # Creating the Confusion Matrix 
    cm = confusion_matrix(y1, y_pred)
    print(cm)
    print(accuracy_score(y1, y_pred))
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
