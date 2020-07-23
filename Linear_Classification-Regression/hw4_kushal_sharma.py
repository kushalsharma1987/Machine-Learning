# Kushal Sharma
# kushals@usc.edu
# HW 4 - Linear Classification, Linear Regression & Logistic Regression
# INF 552 Summer 2020

import math
import random

import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Plot the accuracy graph by calculating percentage of correctly classified points
# out of total points for each iteration.
def plot_accuracy_results(alpha, num_missclassified, n, method):

    fig, ax = plt.subplots()

    iter = np.arange(len(num_missclassified))
    percentage_accuracy = (n - num_missclassified) * 100 / n
    plt.plot(iter, percentage_accuracy)
    plot_title = "Accuracy Graph-" + method
    plt.title(plot_title)
    plt.xlabel('iterations')
    plt.ylabel('Accuracy (# correct classified/total # points)')
    plt.text(.5, .99, 'alpha= ' + str(alpha), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    min_val = num_missclassified.min()
    # Find the index of minimum no. of miss-classified points in all the iterations.
    max_index = np.where(num_missclassified == min_val)[0].tolist()
    max_val = (n - min_val) * 100 / n

    final_iter = len(num_missclassified) - 1
    final_val = (n - num_missclassified[final_iter]) * 100 / n

    print("Maximum accurate classification:", max_val, '% at iteration:', max_index)
    plt.text(.5, .95, 'max.accuracy= ' + str(max_val) + "%", horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
    plt.plot(max_index[-1], max_val, 'o')
    plt.annotate('max='+str(max_val)+"%", xy=(max_index[-1]-5, max_val+3))
    plt.plot(final_iter, final_val, 'o')
    plt.annotate('final=' + str(final_val)+ "%", xy=(final_iter-10, final_val-3))
    ax.set_ylim(ymin=0, ymax=100)
    plt.show()
    fig.savefig(plot_title)


# Plot the graph of number of miss-classified points at each iteration.
def plot_classification_results(alpha, num_missclassified, method):
    fig, ax = plt.subplots()

    iter = np.arange(len(num_missclassified))
    plt.plot(iter, num_missclassified)
    plot_title = "Misclassified Points Graph-" + method
    plt.title(plot_title)
    plt.xlabel('iterations')
    plt.ylabel('# misclassified points')
    plt.text(.5, .99, 'alpha= ' + str(alpha), horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)

    # Find the index of minimum number of miss-classified points in all the iterations.
    min_val = num_missclassified.min()
    min_index = np.where(num_missclassified == min_val)[0].tolist()
    final_iter = len(num_missclassified) - 1
    final_val = num_missclassified[final_iter]
    print("Minimum # of misclassified points:", min_val, 'at iteration:', min_index)

    plt.text(.5, .95, 'min. no. misclass. pts.= ' + str(min_val), horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
    plt.plot(min_index[-1], min_val, 'o')
    plt.annotate('min=' + str(min_val), xy=(min_index[-1] - 5, min_val-100))
    plt.plot(final_iter, final_val, 'o')
    plt.annotate('final=' + str(final_val), xy=(final_iter - 10, final_val + 100))
    ax.set_ylim(ymin=-200, ymax=2000)
    plt.show()
    fig.savefig(plot_title)
    return min_index[-1]


# Plot the In-sample error which is mean of sum of negative log value of score of all the data points in logistic regression
def plot_logistic_results(eta, error_insample, method):

    fig, ax = plt.subplots()

    iter = np.arange(len(error_insample))
    plt.plot(iter, error_insample)
    plot_title = "In-sample Error-" + method
    plt.title(plot_title)
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.text(.95, .95, 'learning rate= ' + str(eta), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    min_val = error_insample.min()
    min_index = np.where(error_insample == min_val)[0].tolist()
    final_iter = len(error_insample) - 1
    final_val = error_insample[final_iter]
    print("Minimum error:", round(min_val, 2), 'at iteration:', min_index)
    plt.plot(min_index[-1], min_val, 'o')
    plt.annotate('min='+str(round(min_val, 2)), xy=(min_index[-1]-.2, min_val-.3))
    plt.plot(final_iter, final_val, 'o')
    plt.annotate('final=' + str(round(final_val, 2)), xy=(final_iter-.5, final_val+.3))
    ax.set_ylim(ymin=0)
    plt.show()
    fig.savefig(plot_title)
    return min_index[-1]


# Plot the points in 3D as well as the hyperplane defined by the hyperplane equation.
def plot_points(points, target, weights, method):

    fig = plt.figure()
    complete_array = np.append(points, target.reshape(-1,1), axis=1)
    positive_label_points = complete_array[complete_array[:,-1] == +1]
    negative_label_points = complete_array[complete_array[:,-1] == -1]

    xp = positive_label_points[:,1]
    yp = positive_label_points[:,2]
    zp = positive_label_points[:,3]

    xn = negative_label_points[:,1]
    yn = negative_label_points[:,2]
    zn = negative_label_points[:,3]

    # create xx, yy and z using hyperplane equation.
    xx, yy = np.meshgrid(range(2), range(2))
    z = (-weights[0] - weights[1] * xx - weights[2] * yy) * 1. / weights[3]

    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z, alpha=0.2, rstride=1, cstride=1, linewidth=0.25)
    # Ensure that the next plot doesn't overwrite the first plot
    ax = plt.gca()

    # ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(xp, yp, zp, c='b', marker='o')
    ax.scatter(xn, yn, zn, c='r', marker='^')
    ax.set_zlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_xlim3d(0, 1)

    plot_title = "3D plot - " + method
    plt.title(plot_title)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(plot_title)
    plt.show()
    # fig.savefig(plot_title)


# Plot the points for Linear Regression
def plot_points_linear(points, weights, method):

    fig = plt.figure()
    xp = points[:, 0]
    yp = points[:, 1]
    zp = points[:, 2]

    # create xx, yy and z using hyperplane equation.
    xx, yy = np.meshgrid(range(3), range(3))
    z = (weights[0] + weights[1] * xx + weights[2] * yy)

    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z, alpha=0.2, rstride=1, cstride=1, linewidth=0.25)
    # Ensure that the next plot doesn't overwrite the first plot
    ax = plt.gca()

    # ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(xp, yp, zp, c='m', marker='o')
    ax.set_zlim3d(0, 5)
    ax.set_ylim3d(0, 1)
    ax.set_xlim3d(0, 1)

    plot_title = "3D plot - " + method
    plt.title(plot_title)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(plot_title)
    plt.show()
    # fig.savefig(plot_title)


# Calculate the updated weights of linear classification problem. If data is linearly separable, this algorithm would
# converge to 0 miss-classified points in limited iterations.
def Linear_classification(nparray, n, d):

    print("Linear Classifier:")
    method = 'Linear Classification'
    weights = np.random.uniform(-100, 100, d+1)
    dict_missclassified = {}
    num_missclassified = n
    points = nparray[:, :d]
    ones = np.ones((n, 1))
    points = np.append(ones, points, axis=1)
    target_y = nparray[:, -1]
    print("Initial Weights:", weights)
    alpha = .1
    iter = 0
    while(num_missclassified > 0):
    # while (iter != 20000 and num_missclassified > 0):
        # After first iteration, update the weights based on randomly selected one of miss-classified point.
        if len(dict_missclassified) != 0:
            x = np.array(dict_missclassified[iter][2])
            y = dict_missclassified[iter][3]
            weights = np.add(weights, y * alpha * x)
            iter += 1
        num_missclassified = 0
        points_violated = []
        for i in range(n):
            p = points[i]
            pred_y = np.sign(np.dot(weights.T, p))
            # Check if the pred_y matches with expected y, if not, consider as miss-classified point.
            if pred_y != target_y[i]:
                num_missclassified += 1
                p = np.append(p, target_y[i])
                points_violated.append(p.tolist())
        # If there is no miss-classified points left, jump to while statement to end the loop.
        if num_missclassified != 0:
            x_violated = random.choice(points_violated)
            y_violated = x_violated[-1]
            x_violated = x_violated[:-1]
            dict_missclassified[iter] = [num_missclassified, weights, x_violated, y_violated]
            if iter % 1000 == 0:
                print(iter, dict_missclassified[iter][0], weights)
        else:
            dict_missclassified[iter] = [num_missclassified, weights, 0, 0]
    # Make numpy array of the number of miss-classified points in all the iterations.
    num_missclassified_all = np.array(list(dict_missclassified.values()))[:, 0]
    # Plot the number of miss-classified points at each iteration and get the index of minimum of it.
    min_index = plot_classification_results(alpha, num_missclassified_all, method)
    # Plot accuracy graph of number of correctly classified points.
    plot_accuracy_results(alpha, num_missclassified_all, n, method)
    # Plot the 3D points along with the hyperplane as decision boundary
    plot_points(points, target_y, weights, method)

    print("Min Iteration:", min_index, dict_missclassified[min_index])
    print("Final Iteration:", iter, dict_missclassified[iter])
    print("====================================\n\n")
    return dict_missclassified[iter][0], dict_missclassified[iter][1]


# Calculate the updated weights of linear classification problem. If data is not linearly separable, this algorithm would
# run till limited iteration (given) and pick the best weight assignment leading to minimum no. of miss-classified points.
def Linear_classification_pocket(nparray, n, d):

    print("Linear Classifier Pocket Algorithm:")
    method = 'Linear Classification (Pocket)'
    weights = np.random.uniform(-100, 100, d + 1)
    dict_missclassified = {}
    num_missclassified = n
    points = nparray[:, :d]
    ones = np.ones((n, 1))
    points = np.append(ones, points, axis=1)
    target_y = nparray[:, -1]
    print("Initial weights:", weights)
    alpha = .1
    iter = 0
    # while(num_missclassified > 0):
    while (iter != 7000 and num_missclassified > 0):
        # After first iteration, update the weights based on randomly selected one of miss-classified point.
        if len(dict_missclassified) > 0:
            x = np.array(dict_missclassified[iter][2])
            y = dict_missclassified[iter][3]
            weights = np.add(weights, y * alpha * x)
            iter += 1
        num_missclassified = 0
        points_violated = []
        for i in range(n):
            p = points[i]
            pred_y = np.sign(np.dot(weights.T, p))
            # Check if the pred_y matches with expected y, if not, consider as miss-classified point.
            if pred_y != target_y[i]:
                num_missclassified += 1
                p = np.append(p, target_y[i])
                points_violated.append(p.tolist())
        # If there is no miss-classified points left, jump to while statement to end the loop.
        if num_missclassified != 0:
            x_violated = random.choice(points_violated)
            y_violated = x_violated[-1]
            x_violated = x_violated[:-1]
            dict_missclassified[iter] = [num_missclassified, weights, x_violated, y_violated]
            if iter % 1000 == 0:
                print(iter, dict_missclassified[iter][0], weights)
        else:
            dict_missclassified[iter] = [num_missclassified, weights, 0, 0]
    # Make numpy array of the number of miss-classified points in all the iterations.
    num_missclassified_all = np.array(list(dict_missclassified.values()))[:, 0]
    # Plot the number of miss-classified points at each iteration and get the index of minimum of it.
    min_index = plot_classification_results(alpha, num_missclassified_all, method)
    # Plot accuracy graph of number of correctly classified points.
    plot_accuracy_results(alpha, num_missclassified_all, n, method)
    # Plot the 3D points along with the hyperplane as decision boundary
    plot_points(points, target_y, weights, method)
    print("Min Iteration:", min_index, dict_missclassified[min_index])
    print("Final Iteration:", iter, dict_missclassified[iter])
    print("====================================\n\n")
    return dict_missclassified[iter][0], dict_missclassified[iter][1]


# Sigmoid function given as = exp(s) / 1 + exp(s)
def sigmoid(s):

    sigmoid_result = np.exp(s) / (1 + np.exp(s))
    return sigmoid_result


# part of derivative of sigmoid function. It's not complete derivative. log function is also associated with it.
def sigmoid_gradient(s):

    sigmoid_grad = 1 / (1 + np.exp(s))
    return sigmoid_grad


# Compute the updated weights on non-linearly separable data using logistic regression. Plot the number of miss-classified
# points, accuracy graph, 3D plot and In-sample error plot.
def logistic_regression(nparray, n, d):

    print("Logistic Regression:")
    method = 'Logistic Regression'
    weights = np.random.uniform(-100, 100, d + 1)
    print("Initial Weights:", weights)
    points = nparray[:, :d]
    ones = np.ones((n, 1))
    points = np.append(ones, points, axis=1)
    target_y = nparray[:, -1]
    eta = .1
    iter = 0
    error_insample_grad = 0
    dict_missclassified = {}
    while(iter != 7000):
        log_score = []
        gradient = []
        num_missclassified = 0
        # Update the weights after first iteration.
        if len(dict_missclassified) > 0:
            weights = weights - eta * error_insample_grad
            iter += 1
        for i in range(n):
            # calculate the score by formula = sigmoid(y*w*x)
            score = sigmoid(target_y[i] * np.dot(weights.T, points[i]))
            # use score probability to find the predicted label
            if score >= 0.5:
                pred_y = +1
            else:
                pred_y = -1
            # Check if predicted label is equal or not to target label. Miss-classify point if not.
            if pred_y != target_y[i]:
                num_missclassified += 1
            # Take negative log of score to convert product of score to addition of log of score of all the points.
            log_score.append(-1 * np.log(score))
            # Calculate the derivative of sigmoid function given as = -y.x / (1 + exp(y.w.x))
            sig_grad = sigmoid_gradient(target_y[i] * np.dot(weights.T, points[i]))
            grad_desc = -1 * sig_grad * target_y[i] * points[i]
            gradient.append(grad_desc)

        # In-sample error = mean of sum of negative log of score of all the points.
        error_insample = np.mean(log_score)
        gradient = np.array(gradient)
        # Graditent of in-sample-error = mean of sum of gradient of all the data points in each dimension.
        error_insample_grad = np.mean(gradient, axis=0)
        dict_missclassified[iter] = [num_missclassified, weights, error_insample]
        if iter % 1000 == 0:
            print(iter, num_missclassified, weights, error_insample)

    # Make numpy array of the number of miss-classified points in all the iterations.
    num_missclassified_all = np.array(list(dict_missclassified.values()))[:, 0]
    # Plot the number of miss-classified points at each iteration and get the index of minimum of it.
    min_index = plot_classification_results(eta, num_missclassified_all, method)
    # Plot accuracy graph of number of correctly classified points.
    plot_accuracy_results(eta, num_missclassified_all, n, method)
    # Plot the 3D points along with the hyperplane as decision boundary
    plot_points(points, target_y, weights, method)
    # Plot the in-sample error by each iteration.
    in_sample_error = np.array(list(dict_missclassified.values()))[:, 2]
    plot_logistic_results(eta, in_sample_error, method)
    print("Min Iteration:", min_index, dict_missclassified[min_index])
    print("Final Iteration:", iter, dict_missclassified[iter])
    print("====================================\n\n")
    return dict_missclassified[iter][0], dict_missclassified[iter][1]


# Calculate the root mean square error of points in linear regression.
def squared_error(points, weights, target):

    points_weights = np.dot(points, weights)
    error_difference = points_weights - target
    square_error_difference = np.square(error_difference)
    root_mean_square_error = np.sqrt(np.mean(square_error_difference))
    print("Root Mean Square Error:", root_mean_square_error)


# Calculate the updated weights for linear regression problem from the formula W = inverse(D.D).D.y
def linear_regression(nparray, n, d):

    print("Linear Regression:")
    points = nparray[:, :d]
    ones = np.ones((n, 1))
    points = np.append(ones, points, axis=1)
    target_z = nparray[:, -1]
    points_square = np.dot(points.T, points)
    points_target = np.dot(points.T, target_z)
    weights = np.dot(np.linalg.inv(points_square), points_target)
    print("Weights:", weights)
    squared_error(points, weights, target_z)
    print("====================================\n\n")
    plot_points_linear(nparray, weights, 'Linear Regression')
    return weights


# Linear Regression using sklearn library
def linear_regression_sklearn(nparray, n, d):

    print("Linear Regression Sklearn:")
    from sklearn.linear_model import LinearRegression
    points = nparray[:, :d]
    # ones = np.ones((n, 1))
    # points = np.append(ones, points, axis=1)
    target_z = nparray[:, -1]
    linear_regressor = LinearRegression()
    print(linear_regressor)
    linear_regressor.fit(points, target_z)
    # weights = linear_regressor.coef_
    weights = np.append(linear_regressor.intercept_.tolist(), linear_regressor.coef_.tolist())
    print("Weights:", weights)
    score = linear_regressor.score(points, target_z)
    print("Score", score)
    # squared_error(points, weights, target_z)
    print("====================================\n\n")


# Linear classification using sklearn
def linear_classifier_sklearn(nparray, n, d):

    print("Linear Classifier Sklearn:")
    from sklearn.linear_model import SGDClassifier
    points = nparray[:, :d]
    target_y = nparray[:, -1]
    clf = SGDClassifier()
    print(clf)
    # fit the classifier
    clf.fit(points, target_y)
    weights = np.append(clf.intercept_.tolist(), clf.coef_.tolist())
    print("Weights:", weights)

    ones = np.ones((n, 1))
    points = np.append(ones, points, axis=1)
    num_missclassified = 0
    points_violated = []
    for i in range(n):
        p = points[i]
        pred_y = np.sign(np.dot(weights.T, p))
        if pred_y != target_y[i]:
            num_missclassified += 1
            p = np.append(p, target_y[i])
            points_violated.append(p.tolist())
    print("# Miss-classified points:", num_missclassified)
    print("Accuray:", (n - num_missclassified) * 100 / n, "%")
    print("====================================\n\n")


# Logistic regression using Sklearn
def logistic_regression_sklearn(nparray, n, d):

    print("Logistic Regression Sklearn:")
    from sklearn.linear_model import LogisticRegression
    points = nparray[:, :d]
    target_y = nparray[:, -1]
    logistic_regressor = LogisticRegression()
    print(logistic_regressor)
    # fit the classifier
    logistic_regressor.fit(points, target_y)
    print("Score:", logistic_regressor.score(points, target_y))
    ones = np.ones((n, 1))
    points = np.append(ones, points, axis=1)
    weights = np.append(logistic_regressor.intercept_.tolist(), logistic_regressor.coef_.tolist())
    print("Weights:", weights)
    num_missclassified = 0
    points_violated = []
    for i in range(n):
        p = points[i]
        pred_y = np.sign(np.dot(weights.T, p))
        if pred_y != target_y[i]:
            num_missclassified += 1
            p = np.append(p, target_y[i])
            points_violated.append(p.tolist())
    print("# Miss-classified points:", num_missclassified)
    print("Accuray:", (n - num_missclassified) * 100 / n, "%")
    print("====================================\n\n")


def main():

    out_file = open("output.txt", "w")

    classification_df = pd.read_csv('classificification.txt', sep=',', header=None)
    d = classification_df.shape[1] - 2
    n = classification_df.shape[0]

    linear_class_perceptron_nparray = classification_df.drop(classification_df.columns[4], axis=1).to_numpy()
    num_missclassified, weights = Linear_classification(linear_class_perceptron_nparray, n, d)
    out_file.write("Linear Classification:\n")
    out_file.write("Weights: " + str(weights) + '\n')
    out_file.write("Accuracy: " + str((n - num_missclassified) * 100 / n) + '%\n')
    out_file.write("=========================================\n\n")

    linear_class_pocket_nparray = classification_df.drop(classification_df.columns[3], axis=1).to_numpy()
    num_missclassified, weights = Linear_classification_pocket(linear_class_pocket_nparray, n, d)
    out_file.write("Linear Classification(Pocket Algorithm):\n")
    out_file.write("Weights: " + str(weights) + '\n')
    out_file.write("Accuracy: " + str((n - num_missclassified) * 100 / n) + '%\n')
    out_file.write("=========================================\n\n")

    logistic_nparray = classification_df.drop(classification_df.columns[3], axis=1).to_numpy()
    num_missclassified, weights = logistic_regression(logistic_nparray, n, d)
    out_file.write("Logistic Regression:\n")
    out_file.write("Weights: " + str(weights) + '\n')
    out_file.write("Accuracy: " + str((n - num_missclassified) * 100 / n) + '%\n')
    out_file.write("=========================================\n\n")

    linear_classifier_sklearn(linear_class_perceptron_nparray, n, d)
    logistic_regression_sklearn(logistic_nparray, n, d)

    regression_df = pd.read_csv('linear-regression.txt', sep=',', header=None)
    d = regression_df.shape[1] - 1
    n = regression_df.shape[0]
    linear_regression_nparray = regression_df.to_numpy()
    weights = linear_regression(linear_regression_nparray, n, d)
    out_file.write("Linear Regression:\n")
    out_file.write("Weights: " + str(weights) + '\n')
    out_file.write("=========================================\n\n")

    linear_regression_sklearn(linear_regression_nparray, n, d)

    out_file.close()

if __name__ == '__main__':
    main()