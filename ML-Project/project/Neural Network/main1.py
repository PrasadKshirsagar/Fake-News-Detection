import numpy as np
from scipy.special import expit as sgm
from matplotlib import pyplot as plt
import cv2
import sys


def import_data():
    global dataset
    global dataset1
    url = "dataset1.txt"
    url1 = "dataset2.txt"
    # names = ['Feature 1', 'Feature 2', 'class']
    # dataset = pandas.read_csv(url, sep=" ")
    dataset = np.loadtxt(url, dtype=np.uint8)
    dataset1 = np.loadtxt(url1, dtype=np.uint8)
    # print(dataset)


#In-built function
def sigmoid(x):
    fx = sgm(x)
    return fx

#Sigmoid-Derivative
def derivatives_sigmoid(x):
    return x * (1 - x)

#Read data from the images in a vector
def return_data(filepath):
    angle = []
    images = []
    # opening file and reading data
    with open(filepath + '/data.txt', 'r') as file:
        for line in file:
            tokens = line.strip().split()
            imagePath = filepath + tokens[0][1:]
            img = cv2.imread(imagePath)
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_vec = gray_img.reshape(1, 1024)
            images.append(img_vec)
            angle.append(float(tokens[1]))
            # print(line)

    # initializing arrays.
    X = np.zeros((len(images), 1024), dtype=float)
    Y = np.zeros((len(images), 1), dtype=float)

    for i in range(len(images)):
        X[i] = images[i]
        Y[i] = angle[i]

    # Standarizing and Normalizing data...
    for i in range(len(images)):
        max_pixel = np.max(X[i, :])
        min_pixel = np.min(X[i, :])

        X[i, :] = (X[i, :] - min_pixel) / (max_pixel - min_pixel)

    for i in range(1, 1024):
        mean = np.mean(X[:, i])
        std = np.std(X[:, i])
        X[:, i] = (X[:, i] - mean) / std

    return X, Y, len(images)


if __name__ == '__main__':
    # input_data, output_data, size = return_data('./steering')
    import_data()
    np.random.shuffle(dataset)
    np.random.shuffle(dataset1)
    X = dataset[:, 0:5000]
    input_data = np.asarray(X)
    y = dataset[:, 5000]
    X1 = dataset1[0:7000, 0:5000]
    y1 = dataset1[0:7000, 5000]
    y = y[:, None]
    y1 = y1[:, None]
    output_data = np.asarray(y)
    output_data = output_data.reshape((len(output_data), 1))
    input_data1 = np.asarray(X1)
    output_data1 = np.asarray(y1)
    output_data1 = output_data1.reshape((len(output_data1), 1))
    # m, n = y.shape
    # b = np.ones((m, 1))
    # X = np.hstack((b, X))
    # m1, n1 = y1.shape
    # b1 = np.ones((m1, 1))
    # X1 = np.hstack((b1, X1))
    
    # dataset = np.concatenate((input_data, output_data), axis=1)
    # msk = np.random.rand(len(dataset)) < 0.8
    # train = dataset[msk]

    # # Divide dataset in training and validation set
    # validate = dataset[~msk]
    # input_data = train[:, :1024]
    # # print(np.shape(input_data))
    # output_data = train[:, 1024]
    # output_data = output_data.reshape((len(output_data), 1))
    # # print(np.shape(output_data))
    # input_data1 = validate[:, :1024]
    # output_data1 = validate[:, 1024]
    # output_data1 = output_data1.reshape((len(output_data1), 1))



    # Variable initialization
    mini_batch_size = 64
    dropout_prob = 1
    epoch = 100  # Setting training iterations
    lr = 0.01  # Setting learning rate
    input_neurons = input_data.shape[1]  # number of features in data set
    hidden1_neurons = 1000  # number of hidden layers neurons 1
    hidden2_neurons = 100  # number of hidden layers neurons 2
    output_neurons = 1  # number of neurons at output layer
    train_sse = []
    validate_sse = []
    train_mse = []
    validate_mse = []
    epoch_array = []

    # Command line arguments
    arguments = sys.argv[1:]
    count = len(arguments)
    if(count == 3):
        epoch = int(sys.argv[1])
        lr = float(sys.argv[2])
        mini_batch_size = int(sys.argv[3])

    if(count == 4):
        epoch = int(sys.argv[1])
        lr = float(sys.argv[2])
        mini_batch_size = int(sys.argv[3])
        dropout_prob = float(sys.argv[4])

    # print(epoch)
    # print(lr)
    # print(mini_batch_size)
    # print(dropout_prob)

    # weight initialization
    weight_input_hidden1 = np.random.uniform(-0.01, 0.01, size=(input_neurons + 1, hidden1_neurons + 1))
    weight_hidden1_hidden2 = np.random.uniform(-0.01, 0.01, size=(hidden1_neurons + 1, hidden2_neurons + 1))
    weight_hidden2_output = np.random.uniform(-0.01, 0.01, size=(hidden2_neurons + 1, output_neurons))
    weight_input_hidden1[0, :] = 0
    weight_hidden1_hidden2[0, :] = 0
    weight_hidden2_output[0, :] = 0

    
    # print(np.shape(input_data), np.shape
    u1 = np.random.binomial(1, dropout_prob, size=input_data.shape) / dropout_prob
    uu2 = np.random.binomial(1, dropout_prob, size=(len(input_data), hidden1_neurons + 1)) / dropout_prob
    uu3 = np.random.binomial(1, dropout_prob, size=(len(input_data), hidden2_neurons + 1)) / dropout_prob


    for i in range(epoch):
        print("Iteration number ", i + 1)
        # input_data *= u1
        # print(input_data)
        # drop-out initialization
        
        total_iter = len(input_data) / mini_batch_size
        total_iter = int(total_iter)
        k = 0
        index = 0
        train_error = 0
        validate_error = 0

        total = np.concatenate((input_data, output_data), axis=1)
        np.random.shuffle(total)

        input_data = total[:, :5000]
        output_data = total[:, 5000]
        output_data = output_data.reshape((len(output_data), 1))
        # print(np.shape(input_data))
        # print(np.shape(output_data))

        while index <= total_iter:
            # print('index is ', index)
            zero_col = np.ones(shape=(mini_batch_size, 1))
            if len(input_data) - k < mini_batch_size:
                inputX = input_data[k:len(input_data), :]
                u2 = uu2[k:len(input_data), :]
                u3 = uu3[k:len(input_data), :]
                output = output_data[k:len(input_data), :]
                zero_col = np.ones(shape=(len(input_data) - k, 1))
            else:
                inputX = input_data[k:k + mini_batch_size, :]
                u2 = uu2[k:k + mini_batch_size, :]
                u3 = uu3[k:k + mini_batch_size, :]
                output = output_data[k:k + mini_batch_size, :]
                zero_col = np.ones(shape=(mini_batch_size, 1))

            inputX = np.concatenate((zero_col, inputX), axis=1)

            # Forward Propogation
            hidden_layer_input1 = np.dot(inputX, weight_input_hidden1)
            hidden_layer_input1 *= u2
            hiddenlayer_activations = sigmoid(hidden_layer_input1)
            
            hidden_layer_input2 = np.dot(hiddenlayer_activations, weight_hidden1_hidden2)
            hidden_layer_input2 *= u3
            hiddenlayer_activations1 = sigmoid(hidden_layer_input2)
            
            output_layer_input = np.dot(hiddenlayer_activations1, weight_hidden2_output)
            
            # Backpropagation
            output_layer_input = sigmoid(output_layer_input)
            E = (output - output_layer_input)*output_layer_input*(1-output_layer_input)
            v1 = E.dot(weight_hidden2_output.T) * u3
            sigmoid1 = derivatives_sigmoid(hiddenlayer_activations1)
            v2 = v1 * sigmoid1
            delta = hiddenlayer_activations.T.dot(v2)
            v3 = v2.dot(weight_hidden1_hidden2.T) * u2
            sigmoid2 = derivatives_sigmoid(hiddenlayer_activations)
            v4 = v3 * sigmoid2
            delta1 = inputX.T.dot(v4)
            weight_hidden2_output += (hiddenlayer_activations1.T.dot(E) * lr)/10
            weight_hidden1_hidden2 += (delta * lr)/10
            weight_input_hidden1 += (delta1 * lr)/10
            k += mini_batch_size
            index += 1
            train_error += np.sum(np.square(E))

        print('train Error is ', train_error / len(input_data))
        train_sse.append(train_error)
        train_mse.append(train_error / len(input_data))
        epoch_array.append(i + 1)

        total_iter = len(input_data1) / mini_batch_size
        total_iter = int(total_iter)
        k = 0
        index = 0
        validate_error = 0
        while index <= total_iter:
            if len(input_data1) - k < mini_batch_size:
                input = input_data1[k:len(input_data1), :]
                output = output_data1[k:len(input_data1), :]
                zero_col = np.zeros(shape=(len(input_data1) - k, 1))
            else:
                input = input_data1[k:k + mini_batch_size, :]
                output = output_data1[k:k + mini_batch_size, :]
                zero_col = np.zeros(shape=(mini_batch_size, 1))

            input = np.concatenate((zero_col, input), axis=1)

            # Forward Propogation
            hidden_layer_input1 = np.dot(input, weight_input_hidden1)
            hiddenlayer_activations = sigmoid(hidden_layer_input1)
            hidden_layer_input2 = np.dot(hiddenlayer_activations, weight_hidden1_hidden2)
            hiddenlayer_activations1 = sigmoid(hidden_layer_input2)
            output_layer_input = np.dot(hiddenlayer_activations1, weight_hidden2_output)
            output_layer_input = sigmoid(output_layer_input)
            
            # Backpropagation
            E = (output - output_layer_input)*output_layer_input*(1-output_layer_input)
            predicting = output_layer_input
            predicting[predicting>0.5] = 1
            predicting[predicting<=0.5] = 0
            k += mini_batch_size
            index += 1
            validate_error += (E.T.dot(E)) / 2
        print('validate Error is ', validate_error[0][0] / len(input_data1))
        print(predicting)
        print('Testing Accuracy: ', np.mean((predicting == output) * 100))
        validate_sse.append(validate_error[0][0])
        validate_mse.append(validate_error[0][0]/len(input_data1))

    print(epoch_array)
    print(train_sse)
    print(train_mse)
    plt.figure(1)
    plt.plot(epoch_array, train_sse, label='training')
    plt.plot(epoch_array, validate_sse, label='validation')
    plt.xlabel("Epoch Size")
    plt.ylabel("Sum of Squares training and validation error")
    if(dropout_prob == 1):
    	plt.title("SSE on Training and Validation Set.\nLearning rate=%f minibatch = %d No Dropout" %(lr, mini_batch_size))
    else:
    	plt.title("SSE on Training and Validation Set.\nLearning rate=%f minibatch = %d Dropout = %f" %(lr, mini_batch_size, dropout_prob))
    # plt.savefig('f1.png')
    # plt.show()print(epoch)
    # print(lr)
    # print(mini_batch_size)
    

    plt.figure(2)
    plt.plot(epoch_array, train_mse, label='training')
    plt.plot(epoch_array, validate_mse, label='validation')
    plt.xlabel("Epoch Size")
    plt.ylabel("Mean Squares training and validation error")
    if(dropout_prob == 1):
    	plt.title("MSE on Training and Validation Set.\nLearning rate=%f minibatch = %d No Dropout" %(lr, mini_batch_size))
    else:
    	plt.title("MSE on Training and Validation Set.\nLearning rate=%f minibatch = %d Dropout = %f" %(lr, mini_batch_size, dropout_prob))
    # plt.savefig('f2.png')
    plt.show()
