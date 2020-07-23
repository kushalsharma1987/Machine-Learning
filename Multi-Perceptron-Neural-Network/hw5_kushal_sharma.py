# Kushal Sharma
# kushals@usc.edu
# HW 5 - Multi-Layer Perceptron Neural Network
# INF 552 Summer 2020

from os import  path
import numpy as np
import matplotlib.image as image
from matplotlib import pyplot as plt

# Read PGM file in matplotlib and return the normalized numpy array
def read_pgm_matplotlib(filename):

    img_data = image.imread(filename)
    img_arry = np.asarray(img_data).flatten()
    return img_arry / 255


# Sigmoid function
def sigmoid(s):
    return 1 / (1 + np.exp(-1 * s))


# Derivative of sigmoid function
def sigmoid_derivative(s):
    return np.ones_like(s) - np.square(sigmoid(s))


# Perceptron Neural network with iterations on samples instead of using numpy array
def perceptron_neural_network_iteration(X, y):

    inp_dim = X.shape[1]
    inp_sample = X.shape[0]
    level_1_dim = 100
    level_2_dim = 1
    bias = 1
    np.random.seed(100)
    weights_1 = np.random.uniform(-0.01, 0.01, size=(inp_dim + bias, level_1_dim))
    weights_2 = np.random.uniform(-0.01, 0.01, size=(level_1_dim + bias, 1))
    eta = 0.1
    epochs = 1
    print("Iterations:")
    for e in range(epochs):
        for s in range(10):
            ################ Feed forward #####################
            x_inp = np.append(1, X[s]).reshape((inp_dim + bias, 1))
            s_1 = np.zeros((level_1_dim, 1))
            for j in range(level_1_dim):
                for i in range(inp_dim):
                    s_1[j] += weights_1[i][j] * x_inp[i]
            x_1 = np.zeros((level_1_dim, 1))
            for j in range(level_1_dim):
                x_1[j] = sigmoid(s_1[j])
            x_1 = np.append(1, x_1).reshape((x_1.shape[0] + 1, 1))
            s_2 = np.zeros((level_2_dim, 1))
            for k in range(level_2_dim):
                for j in range(level_1_dim+bias):
                    s_2[k] += weights_2[j][k] * x_1[j]
            x_2 = np.zeros((level_2_dim, 1))
            for k in range(level_2_dim):
                x_2[k] = sigmoid(s_2[k])
            error = np.square(x_2 - y[s])

            ################ Back Propagation ########################
            delta_2 = 2 * (x_2 - y[s]) * sigmoid_derivative(s_2)
            derivative_error_weight_2 = np.zeros((level_1_dim + bias, 1))
            for i in range(level_1_dim+bias):
                derivative_error_weight_2[i] = x_1[i] * delta_2
            delta_1 = np.zeros((level_1_dim, 1))
            weight_delta_2_sum = 0
            for i in range(level_1_dim):
                for j in range(level_2_dim):
                    weight_delta_2_sum += weights_2[i+1][j] * delta_2
                delta_1[i] = (1 - np.square(x_1[i+1])) * weight_delta_2_sum
            derivative_error_weight_1 = np.zeros((inp_dim + bias, level_1_dim))
            for i in range(level_1_dim + bias):
                for j in range(level_2_dim):
                    derivative_error_weight_1[i][j] = x_inp[i] * delta_1[j]
            weights_2 = weights_2 - eta * derivative_error_weight_2
            weights_1 = weights_1 - eta * derivative_error_weight_1
            print("Error:", error)


# Train neural network with numpy array and functions.
def nn_numpy_train(X, y):

    # print("Train...")
    inp_dim = X.shape[1]
    num_sample = X.shape[0]
    level_1_dim = 100
    level_2_dim = 1
    bias = 1
    # np.random.seed(100)
    weights_1 = np.random.uniform(-0.01, 0.01, size=(inp_dim + bias, level_1_dim))
    weights_2 = np.random.uniform(-0.01, 0.01, size=(level_1_dim + bias, level_2_dim))
    eta = 0.1
    epochs = 1000
    y = y.reshape((num_sample, 1))
    input = np.append(X, y, axis=1)
    error_epoch = []
    num_mislabeled_all = []
    for e in range(epochs):
        # Shuffle the input in every epoch so the order is not consistent
        np.random.shuffle(input)
        num_mislabeled = 0
        error = []
        prediction = []
        mislabeled_index = []
        for num, sample in enumerate(input):
            ############## Feed forward #####################
            x_inp = sample[:-1]
            y_inp = sample[-1]
            x_inp = np.append(1, x_inp).reshape((inp_dim + bias, 1))
            s_1 = np.dot(weights_1.T, x_inp)
            x_1 = sigmoid(s_1)
            x_1 = np.append(1, x_1).reshape((level_1_dim + bias, 1))
            s_2 = np.dot(weights_2.T, x_1)
            x_2 = sigmoid(s_2)
            squared_error = np.square(x_2 - y_inp)
            error.append(squared_error)
            if x_2 >= 0.5:
                y_pred = 1
            else:
                y_pred = 0
            if y_pred != y_inp:
                num_mislabeled += 1
            if e+1 == epochs:
                if y_pred != y_inp:
                    mislabeled_index.append(num)
                prediction.append(y_pred)
            ############ Back Propagation #####################
            delta_2 = 2 * (x_2 - y_inp) * sigmoid_derivative(s_2)
            derivative_error_weight_2 = np.dot(x_1, delta_2.T)
            weights_2_without_bias = np.delete(weights_2, obj=0, axis=0)
            # For delta_1, make sure weight bias is not included for Dimension consistency.
            delta_1 = sigmoid_derivative(s_1) * np.dot(weights_2_without_bias, delta_2)
            derivative_error_weight_1 = np.dot(x_inp, delta_1.T)
            weights_2 = weights_2 - eta * derivative_error_weight_2
            weights_1 = weights_1 - eta * derivative_error_weight_1
        print("Epoch:", e+1, "Mislabeled:", num_mislabeled, "Squared Error:", np.mean(error))
        num_mislabeled_all.append(num_mislabeled)
        error_epoch.append(np.mean(error))
    return num_mislabeled_all, mislabeled_index, error_epoch, prediction, weights_1, weights_2


# Test/Predict Neural Network with numpy array and functions.
def nn_numpy_test(X, y, weights_1, weights_2):

    # print("Test...")
    inp_dim = X.shape[1]
    num_sample = X.shape[0]
    level_1_dim = 100
    level_2_dim = 1
    bias = 1
    y = y.reshape((num_sample, 1))
    input = np.append(X, y, axis=1)
    prediction = []
    num_mislabeled = 0
    mislabeled_index = []
    error = []
    for num, sample in enumerate(input):
        ############## Feed forward #####################
        x_inp = sample[:-1]
        y_inp = sample[-1]
        x_inp = np.append(1, x_inp).reshape((inp_dim + bias, 1))
        s_1 = np.dot(weights_1.T, x_inp)
        x_1 = sigmoid(s_1)
        x_1 = np.append(1, x_1).reshape((level_1_dim + bias, 1))
        s_2 = np.dot(weights_2.T, x_1)
        x_2 = sigmoid(s_2)
        squared_error = np.square(x_2 - y_inp)
        error.append(squared_error)
        if x_2 >= 0.5:
            y_pred = 1
        else:
            y_pred = 0
        if y_pred != y_inp:
            num_mislabeled += 1
            mislabeled_index.append(num)
        prediction.append(y_pred)
    error_epoch = np.mean(error)
    return num_mislabeled, mislabeled_index, error_epoch, prediction


# Plot the accuracy graph for training samples per epoch and for Test samples once.
def plot_accuracy(accuracy_train, accuracy_test):

    fig, ax = plt.subplots()
    epochs = np.arange(len(accuracy_train))
    plt.plot(epochs, accuracy_train)
    # plt.annotate('train acc:' + str(round(accuracy_train[-1], 2)) + '%', xy=(len(accuracy_train), accuracy_train[-1] + 5))
    plt.scatter(epochs[-1], accuracy_test, c='r')
    # plt.annotate('test acc:' + str(round(accuracy_test, 2)) + '%', xy=(0, accuracy_test+5))
    plt.text(.5, .10, 'Train Accuracy: ' + str(round(accuracy_train[-1], 2)) + '%', horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes)
    plt.text(.5, .05, 'Test Accuracy: ' + str(round(accuracy_test, 2)) + '%', horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes)
    plt.title('Train-Test Accuracy')
    plt.ylabel('accuracy(%)')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    # plt.ylim((0, 100))
    # plt.grid(True)
    plt.show()
    fig.savefig("Train-Test Accuracy Graph")


# Plot the error value for training samples per epochs and test samples.
def plot_error(error_train, error_test):

    fig, ax = plt.subplots()
    epochs = np.arange(len(error_train))
    plt.plot(epochs, error_train)
    # plt.plot(error_test, 'rp', markersize=10)
    plt.scatter(epochs[-1], error_test, color='r')
    plt.text(.5, .95, 'Train Error: ' + str(round(error_train[-1], 2)), horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes)
    plt.text(.5, .90, 'Test Error: ' + str(round(error_test, 2)), horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes)
    plt.title('Train-Test Error')
    plt.ylabel('Error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    # plt.grid(True)
    plt.show()
    fig.savefig("Train-Test Error Graph")


# Neural network by using sklearn libraries.
def nn_sklearn(train_image, train_label, test_image, test_label):

    print("Sklearn:\n")
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='sgd', learning_rate_init=0.1, hidden_layer_sizes=(100), max_iter=1000)
    print(clf)
    clf.fit(train_image, train_label)
    predict_train = clf.predict(train_image)
    predict_test = clf.predict(test_image)
    num_mislabeled = 0
    mislabeled_index = []
    num_test_samples = len(test_label)
    for i in range(num_test_samples):
        if predict_test[i] != test_label[i]:
            num_mislabeled += 1
            mislabeled_index.append(i)

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(test_label, predict_test))
    print(classification_report(test_label, predict_test))

    return num_mislabeled, mislabeled_index, predict_test


# Main function
def main():

    train_images = list()
    train_label = list()
    train_file = open("downgesture_train.list", "r")
    train_images_name = []
    for filename in train_file:
        if path.exists(filename.rstrip()):
            train_images_name.append(filename.rstrip())
            image_array = read_pgm_matplotlib(filename.rstrip())
            train_images.append(image_array)
            if filename.find('down') != -1:
                train_label.append(1)
            else:
                train_label.append(0)
    train_file.close()
    train_images = np.array(train_images)
    train_label = np.array(train_label)
    num_mislabeled_train, mislabeled_index_train, error_train, prediction_train, train_weights_1, train_weights_2 \
        = nn_numpy_train(train_images, train_label)
    num_train_samples = len(train_images)
    accuracy_train = [(num_train_samples - x) * 100/num_train_samples for x in num_mislabeled_train]

    test_images = list()
    test_label = list()
    test_images_name = []
    test_file = open("downgesture_test.list", "r")
    for filename in test_file:
        if path.exists(filename.rstrip()):
            test_images_name.append(filename.rstrip())
            image_array = read_pgm_matplotlib(filename.rstrip())
            test_images.append(image_array)
            if filename.find('down') != -1:
                test_label.append(1)
            else:
                test_label.append(0)
    test_file.close()
    test_images = np.array(test_images)
    test_label = np.array(test_label)
    num_test_samples = len(test_images)
    num_mislabeled_test, mislabeled_index_test, error_test, prediction_test = nn_numpy_test(test_images, test_label, train_weights_1, train_weights_2)
    accuracy_test = (num_test_samples - num_mislabeled_test) * 100 / num_test_samples

    output = open('output.txt', 'w')
    print("Training mislabeled(final epoch):", num_mislabeled_train[-1], "/", num_train_samples)
    print("Training Accuracy(final epoch):", accuracy_train[-1], "%")
    print("Training Squared Error(final epoch):", error_train[-1])
    print("Training Prediction(final epoch): entries marked with * are mis-labeled")
    output.write("Training mislabeled(final epoch): " + str(num_mislabeled_train[-1]) + "/" + str(num_train_samples) + '\n')
    output.write("Training Accuracy(final epoch): " + str(accuracy_train[-1]) + "%\n")
    output.write("Training Error(final epoch): " + str(error_train[-1]) + "\n")
    output.write("Training Prediction(final epoch): entries marked with * are mis-labeled" + '\n')
    for i in range(num_train_samples):
        index = str(i+1) + "."
        if i in mislabeled_index_train:
            print(index, train_images_name[i], "=>", prediction_train[i], "*")
            output.write(index + " " + train_images_name[i] + " => " + str(prediction_train[i]) + " * " + '\n')
        else:
            print(index, train_images_name[i], "=>", prediction_train[i])
            output.write(index + " " + train_images_name[i] + " => " + str(prediction_train[i]) + '\n')
    print("==============================================\n\n")
    output.write("==============================================\n\n")

    print("Test mislabeled:", num_mislabeled_test, "/", num_test_samples)
    print("Test Accuracy:", accuracy_test, "%")
    print("Test Squared Error:", error_test)
    print("Test Prediction: entries marked with * are mis-labeled")
    output.write("Test mislabeled: " + str(num_mislabeled_test) + "/" + str(num_test_samples) + '\n')
    output.write("Test Accuracy: " + str(accuracy_test) + "%\n")
    output.write("Test Error: " + str(error_test) + "\n")
    output.write("Test Prediction: entries marked with * are mis-labeled" + '\n')
    for i in range(num_test_samples):
        index = str(i+1) + "."
        if i in mislabeled_index_test:
            print(index, test_images_name[i], "=>", prediction_test[i], "*")
            output.write(index + " " + test_images_name[i] + " => " + str(prediction_test[i]) + " * " + '\n')
        else:
            print(index, test_images_name[i], "=>", prediction_test[i])
            output.write(index + " " + test_images_name[i] + " => " + str(prediction_test[i]) + '\n')
    print("==============================================\n\n")
    output.write("==============================================\n\n")
    output.close()

    plot_accuracy(accuracy_train, accuracy_test)
    plot_error(error_train, error_test)

    # Sklearn
    num_mislabeled_sklearn, mislabeled_index_sklearn, prediction_sklearn = nn_sklearn(train_images, train_label, test_images, test_label)
    accuracy_sklearn = (num_test_samples - num_mislabeled_sklearn ) * 100 / num_test_samples
    print("Sklearn mislabeled:", num_mislabeled_sklearn)
    print("Sklearn Accuracy:", accuracy_sklearn, "%")
    print("Sklearn Prediction: entries marked with * are mis-labeled")
    for i in range(num_test_samples):
        index = str(i + 1) + "."
        if i in mislabeled_index_sklearn:
            print(index, test_images_name[i], "=>", prediction_sklearn[i], "*")
        else:
            print(index, test_images_name[i], "=>", prediction_sklearn[i])


if __name__ == '__main__':
    main()