import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy._lib.six import xrange

MARKERS = ['+', 'x', '.']
COLORS = ['red', 'green', 'blue']
epsilon = .00000000000000000000001

cost_array = []
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


def plot_dataset():
    global dataset
    sns.lmplot('Feature 1', 'Feature 2', dataset, hue='class', fit_reg=False)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    fig.savefig('11.png')
    plt.show()


# Return Sigmoid of z
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def func(theta, X):
    theta = theta[:, None]
    return sigmoid(X.dot(theta))


# Return cost of given theta for logistic regression
def regularised_cost(theta, X, y, lamda):
    val = func(theta, X)
    val[val < epsilon] = epsilon  # values close to zero are set to tol
    val[(val < 1 + epsilon) & (val > 1 - epsilon)] = 1 - epsilon  # values close to 1 get set to 1 - tol
    lmbda_cont = (float(lamda) / 2) * theta ** 2
    cost = y * np.log(val) + (-y + 1) * np.log(-val + 1)
    m = len(y) * 1.0
    J = -sum(cost) / m + sum(lmbda_cont[1:]) / m
    return J[0]


# Return gradient of theta
def gradient(theta, X, y, lamda):
    value = func(theta, X)
    l = len(y)
    lamda_grad = float(lamda) * theta / l
    grad = (value - y).T.dot(X) / l + lamda_grad.T
    # grad = X.T.dot(value-y)/l + lamda_grad
    grad[0][0] -= lamda_grad[0]
    # print(grad)
    # return grad
    return np.ndarray.flatten(grad)


# predicts the output as 0 or 1
def predict(theta, X):
    h_theta = func(theta, X)
    return np.round(h_theta)


def sigmoid1(X):
    return 1 / (1 + np.exp(-X))


def predict1(X, theta):
    X = np.insert(X, 0, np.ones(len(X)), axis=1)
    return (sigmoid1(X * np.matrix(theta).T) >= 0.5).astype(int)


# optimization using gradient descent
def optimise_grad_descent(alpha, num_iter, X, initial_theta, y, lambda2):
    global cost_array	
    theta = initial_theta
    for i in range(num_iter):
        grad = gradient(theta, X, y, lambda2)
        theta -= alpha * grad
        cost = regularised_cost(theta, X, y, lambda2)
        cost_array.append(cost)
    return theta


# optimization using Newton-Raphson method
def newton_method(theta, X, y, lambdapar, n):
    theta = theta.reshape(n)
    value = func(theta, X)
    l = len(y)
    lamda_grad = float(lambdapar) * theta / l
    # Grad using lambda value
    gradll = (value - y).T.dot(X) / l + lamda_grad.T
    ds = value * (1 - value)
    # Diagonal matrix 'R'
    D = np.diag(ds.ravel())
    # identity matrix
    idty = np.identity(theta.shape[0])
    # Hessian matrix
    H = X.T.dot(D.dot(X)) - lambdapar * idty
    # Calculating Hessian inverse
    Hinv = np.linalg.inv(H)
    theta = theta.reshape(n, 1)
    # updating theta
    theta = theta - Hinv.dot(gradll.T)
    return theta


# Super-fuction which controls newton raphson method
def optimise_newton_raphson(num_iter, X, initial_theta, y, lambda2, n):
    theta = initial_theta
    for i in range(num_iter):
        val = newton_method(theta, X, y, lambda2, n)
        theta = val
        cost = regularised_cost(theta, X, y, lambda2)
        cost_array.append(cost)
    theta = theta.reshape(n)
    return theta


# feature transform for higher dimension features
def featuretransform(X1, X2, degree=2):
    num_features = (degree + 1) * (degree + 2) // 2
    out = np.empty(shape=(X1.shape[0], num_features), dtype=float)
    k = 0
    for i in xrange(0, degree + 1):
        for j in xrange(i + 1):
            new_feature_values = X1 ** (i - j) * X2 ** j
            out[:, k] = new_feature_values[:, 0]
            k += 1
    return out


def plot_decision_boundary(theta, X, y, lmbda):
    sns.lmplot('Feature 1', 'Feature 2', hue='class', data=dataset, fit_reg=False, height=6, aspect=1.5, palette='Dark2', legend=False)
    z = np.zeros([50, 50])
    uu = np.linspace(-2, 8.0, 50)
    vv = np.linspace(-2, 8.0, 50)
    for i, u in enumerate(uu):
        for j, v in enumerate(vv):
            # z[i, j] = np.dot(np.array([[1,u,v]]), theta)[0]
            z[i, j] = np.dot(featuretransform(np.array([[u]]), np.array([[v]])), theta)[0]
    plt.contour(uu, vv, z.T, [0], colors='dodgerblue')
    plt.axis([-2, 8, -2, 8])
    plt.xticks(np.arange(-2, 8, .5))
    plt.yticks(np.arange(-2, 8, .5))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(('y=1', 'y=0'), loc='upper right', numpoints=1)
    plt.title('Decision boundary for lambda = ' + str(lmbda), fontsize=13)
    plt.show()


def plot_decision_boundary1(theta, X, y, lmbda):
    sns.lmplot('Feature 1', 'Feature 2', hue='class', data=dataset, fit_reg=False, height=6, aspect=1.5, palette='Dark2', legend=False)
    z = np.zeros([50, 50])
    uu = np.linspace(-2, 8.0, 50)
    vv = np.linspace(-2, 8.0, 50)
    for i, u in enumerate(uu):
        for j, v in enumerate(vv):
            z[i, j] = np.dot(np.array([[1,u,v]]), theta)[0]
            # z[i, j] = np.dot(featuretransform(np.array([[u]]), np.array([[v]])), theta)[0]
    plt.contour(uu, vv, z.T, [0], colors='dodgerblue')
    plt.axis([-2, 8, -2, 8])
    plt.xticks(np.arange(-2, 8, .5))
    plt.yticks(np.arange(-2, 8, .5))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(('y=1', 'y=0'), loc='upper right', numpoints=1)
    plt.title('Decision boundary for lambda = ' + str(lmbda), fontsize=13)
    plt.show()


def main():
    import_data()
    global cost_array
    # plot_dataset()
    np.random.shuffle(dataset)
    np.random.shuffle(dataset1)
    # print(dataset)
    X = dataset[:, 0:200]
    y = dataset[:, 200]
    X1 = dataset1[0:1000, 0:200]
    y1 = dataset1[0:1000, 200]
    y = y[:, None]
    y1 = y1[:, None]
    m, n = y.shape
    b = np.ones((m, 1))
    X = np.hstack((b, X))
    m1, n1 = y1.shape
    b1 = np.ones((m1, 1))
    X1 = np.hstack((b1, X1))
    m, n = X.shape
    print(m, n)
    initial_theta = np.zeros(n)
    lambda2 = 2    #1 #0.01 #underfit (10, 10)#overfit(0.01, 410) #fit(0.01, 100)
    alpha = 1       #1
    num_iter = 1000       #100000 #100  #410
    cost = regularised_cost(initial_theta, X, y, lambda2)
    cost_array.append(cost)
    print('Gradient descent working....')
    print('Initial theta is ', initial_theta)
    print('Initial cost is ', cost)
    theta = optimise_grad_descent(alpha, num_iter, X, initial_theta, y, lambda2)
    print('Final theta is ', theta)
    cost = regularised_cost(theta, X, y, lambda2)
    print('Final cost is ', cost)
    cost_array.append(cost)
    p = predict(theta, X)
    print('Training Accuracy: ', np.mean((p == y) * 100))
    p = predict(theta, X1)
    print('Testing Accuracy: ', np.mean((p == y1) * 100))
    num_array = []
    for i in range(len(cost_array)):
    	num_array.append(i)
    print(num_array)
    plt.plot(num_array, cost_array)
    plt.xlabel('Iteration#')
    plt.ylabel('Cost')
    plt.title('L2 Logistic Regression')
    plt.show()
    # plot_decision_boundary1(theta, X, y, lambda2)
    #
    # print('Newton Raphson working....')
    # theta = optimise_newton_raphson(num_iter, X, initial_theta, y, lambda2, n)
    # print('final theta is ', theta)
    # cost = regularised_cost(theta, X, y, lambda2)
    # print('final cost is ', cost)
    # p = predict(theta, X)
    # print('Training Accuracy: ', np.mean((p == y) * 100))
    # plot_decision_boundary1(theta, X, y, lambda2)
    #
    # X1 = X[:, 1][:, None]
    # X2 = X[:, 2][:, None]
    # X = featuretransform(X1, X2, 2)
    # m, n = X.shape
    # initial_theta = np.zeros(n)
    # print('Feature Transformation plus Newton Raphson....')
    # print('Initial theta is ', initial_theta)
    # print('Cost is ', cost)
    # theta = optimise_newton_raphson(num_iter, X, initial_theta, y, lambda2, n)
    # print('Final theta is ', theta)
    # cost = regularised_cost(theta, X, y, lambda2)
    # print('Final cost is ', cost)
    # p = predict(theta, X)
    # print('Training Accuracy: ', np.mean((p == y) * 100))
    # plot_decision_boundary(theta, X, y, lambda2)


if __name__ == '__main__':
    main()
