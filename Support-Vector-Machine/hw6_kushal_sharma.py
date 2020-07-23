# Kushal Sharma
# kushals@usc.edu
# HW 6 - Support Vector Machine
# INF 552 Summer 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2):
    return (np.dot(x1, x2)) ** 2


def quadratic_prog_solver(q_matrix, yn, N):
    '''
    General Quadratic Program can be expressed as follows:
            minimize (1/2)x.T P x + q.T x + r
            subject to G x <= h
                        A x = b
    The objective function we have for SVM is as follows:
            minimize (1/2) alpha.T Q_matrix alpha - alpha(1) - alpha(2) - .... - alpha(N)
            subject to  alpha(1) + alpha(2) + ... + alpha(N) > 0
                        [alpha(1)*label(1) + alpha(2)*label(2) + ... + alpha(N)*label(N)] = 0
    '''
    P = matrix(q_matrix)
    q = matrix(-1 * np.ones((N, 1)))
    G = matrix(np.diag(np.ones(N) * -1))
    h = matrix(np.zeros((N, 1)))
    A = matrix(yn, (1, N))
    b = matrix(0.0)
    # Quadratic Programming solver by cvxopt
    solution = solvers.qp(P, q, G, h, A, b)
    # Lagrangian multiplier
    alpha = np.ravel(solution['x'])
    # Support vectors with lagrangian multiplier > 0
    support_vector = alpha > 1e-5
    index = []
    for i, value in enumerate(support_vector):
        if value:
            index.append(i)
    # shortcut way to find the index of true in support_vector
    # ind = np.arange(len(alpha))[support_vector]
    # print("Support alpha:", alpha[support_vector])
    # print("Support index:", index)
    return alpha, index


# Plot the linearly separable points and the hyperplane defined by the hyperplane equation in 2D space.
def plot_linsep_points(points, target, w=0, b=0, points_sv=0):

    fig = plt.figure()
    complete_array = np.append(points, target.reshape(-1,1), axis=1)
    positive_label_points = complete_array[complete_array[:,-1] == +1]
    negative_label_points = complete_array[complete_array[:,-1] == -1]
    xp = positive_label_points[:,0]
    yp = positive_label_points[:,1]

    xn = negative_label_points[:,0]
    yn = negative_label_points[:,1]

    plt.scatter(xp, yp, c='b')
    plt.scatter(xn, yn, c='r')

    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c;   w = [w0 w1];   x = [x0 x1]
        return (-w[0] * x - b + c) / w[1]

    # w.x + b = 0
    a0 = -1
    a1 = f(a0, w, b)
    b0 = 1
    b1 = f(b0, w, b)
    plt.plot([a0, b0], [a1, b1], "k")

    # w.x + b = 1
    a0 = -1;
    a1 = f(a0, w, b, 1)
    b0 = 1;
    b1 = f(b0, w, b, 1)
    plt.plot([a0, b0], [a1, b1], "k--")

    # w.x + b = -1
    a0 = -1;
    a1 = f(a0, w, b, -1)
    b0 = 1;
    b1 = f(b0, w, b, -1)
    plt.plot([a0, b0], [a1, b1], "k--")

    # mark the support vectors in green
    x_sv = points_sv[:,0]
    y_sv = points_sv[:,1]
    plt.scatter(x_sv, y_sv, c='g')

    plot_title = "Linearly Separable Point Plot"
    plt.title(plot_title)
    plt.xlabel('X Label')
    plt.ylabel('Y Label')

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    plt.show()
    fig.savefig(plot_title)

# Plot the non-linearly separable points and the hyperplane defined by the hyperplane equation in 2D space.
def plot_nonlinsep_points(points, target, alpha=0, index=0, b=0, points_sv=0):

    fig = plt.figure()
    complete_array = np.append(points, target.reshape(-1,1), axis=1)
    positive_label_points = complete_array[complete_array[:,-1] == +1]
    negative_label_points = complete_array[complete_array[:,-1] == -1]
    xp = positive_label_points[:,0]
    yp = positive_label_points[:,1]

    xn = negative_label_points[:,0]
    yn = negative_label_points[:,1]

    plt.scatter(xp, yp, c='b')
    plt.scatter(xn, yn, c='r')

    X1, X2 = np.meshgrid(np.linspace(-25, 25, 100), np.linspace(-25, 25, 100))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    y_predict = np.zeros(len(X))
    for j in range(len(X)):
        s = 0
        for i in index:
            s += alpha[i] * target[i] * polynomial_kernel(points[i], X[j].T)
        y_predict[j] = s
    Z = (y_predict + b).reshape(X1.shape)

    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    x_sv = points_sv[:,0]
    y_sv = points_sv[:,1]
    plt.scatter(x_sv, y_sv, c='g')

    plot_title = "Non-Linearly Separable Points Plot"
    plt.title(plot_title)
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.legend(['+1', '-1'], loc='upper right')

    plt.show()
    fig.savefig(plot_title)


def svm_linsep(linsep_points, linsep_labels, outfile):

    print('Linearly Separable Case:')
    N, d = linsep_points.shape

    # q_matrix_iter = np.ones((N, N))
    # for i in range(len(linsep_labels)):
    #     for j in range(len(linsep_labels)):
    #         kernal_function = linear_kernel(linsep_points[i], linsep_points[j].T)
    #         q_matrix_iter[i][j] = linsep_labels[i] * linsep_labels[j] * dot_points
    # print('iteration:', q_matrix_iter)

    linsep_kernel = linear_kernel(linsep_points, linsep_points.T)
    linsep_q_matrix = np.outer(linsep_labels, linsep_labels) * linsep_kernel

    alpha, index = quadratic_prog_solver(linsep_q_matrix, linsep_labels, N)

    points_support_vectors = []
    labels_support_vectors = []
    weights = np.zeros(d)
    for i in index:
        points_support_vectors.append(linsep_points[i])
        labels_support_vectors.append(linsep_labels[i])
        weights += alpha[i] * linsep_labels[i] * linsep_points[i]
    # bias term calculation for one of the support vector.
    bias = linsep_labels[index[0]] - np.dot(weights, linsep_points[index[0]].T)

    print('Linear kernel function:', 'X1.X2')
    print('Support vectors indices:', index)
    print('Support lagrangian multiplier:', alpha[index])
    support_vectors = np.append(np.array(points_support_vectors), np.array(labels_support_vectors).reshape(-1, 1),
                                axis=1)
    print('Support Vectors:\n', support_vectors)
    print('weights:', weights)
    print('bias:', bias)
    equation_line = str(round(weights[0], 2)) + "x " + ["", "+"][weights[1] >= 0] + str(round(weights[1], 2)) + "y " + \
                    ["", "+"][bias >= 0] + str(round(bias, 2))
    print('Equation of line:', equation_line, '= 0')

    # Plot the points and the hyperline/hyperplane.
    plot_linsep_points(linsep_points, linsep_labels, weights, bias, np.array(points_support_vectors))

    y_predict = np.zeros(len(linsep_labels))
    misclassified = 0
    # for j in range(len(X)):
    for j in range(len(linsep_points)):
        s = 0
        for i in index:
            s += alpha[i] * linsep_labels[i] * linear_kernel(linsep_points[i], linsep_points[j].T)
        y_predict[j] = np.sign(s + bias)
        if y_predict[j] != linsep_labels[j]:
            misclassified += 1
    print('Accuracy:', (N - misclassified) * 100/N, '%')
    print('=============================================\n\n')

    outfile.write('Linearly Separable Case:\n')
    outfile.write('Linear kernel function: ' + 'X1.X2\n')
    outfile.write('Support vectors indices: ' + str(index) + '\n')
    outfile.write('Support lagrangian multiplier: ' + str(alpha[index]) + '\n')
    outfile.write('Support Vectors:\n' + str(support_vectors) + '\n')
    outfile.write('weights: ' + str(weights) + '\n')
    outfile.write('bias: ' + str(bias) + '\n')
    outfile.write('Equation of line: ' + str(equation_line) + '= 0\n')
    outfile.write('Accuracy: ' + str((N - misclassified) * 100/N) + '%\n')
    outfile.write('=============================================\n\n')


def svm_nonlinsep(nonlinsep_points, nonlinsep_labels, outfile):

    print('Non-linearly Separable Case:')
    N, d = nonlinsep_points.shape

    # nonlinsep_q_matrix_iter = np.ones((N, N))
    # for i in range(len(nonlinsep_labels)):
    #     for j in range(len(nonlinsep_labels)):
    #         kernel_function = polynomial_kernel(nonlinsep_points[i], nonlinsep_points[j].T)
    #         nonlinsep_q_matrix_iter[i][j] = nonlinsep_labels[i] * nonlinsep_labels[j] * kernel_function
    # print("q_iter:", nonlinsep_q_matrix_iter)

    nonlinsep_kernel = polynomial_kernel(nonlinsep_points, nonlinsep_points.T)
    nonlinsep_q_matrix = np.outer(nonlinsep_labels, nonlinsep_labels) * nonlinsep_kernel

    alpha, index = quadratic_prog_solver(nonlinsep_q_matrix, nonlinsep_labels, N)

    points_support_vectors = []
    labels_support_vectors = []
    '''
    Two examples of polynomial functions. We are using second one.
    1. 
            (1+x1x1'+x2x2') ** 2 = 1 + (x1x1')**2 + (x2x2')**2 + 2x1x1' + 2x2x2' + 2x1x2x1'x2'
            -(x2x2')**2 - 2x2x2' - 2x1x2x1'x2' = 1 + (x1x1')**2 + 2x1x1'
            1 - 2x1x1'x2x2' - (1 + x2x2')**2 = (1 + x1x1')**2 
    2.
            (x1x1'+x2x2')**2 = (x1x1')**2 + (x2x2')**2 + 2x1x1'x2x2'
    '''

    # Initialize the bias term with ym as b = ym - sum(an*yn*xn*xm) for varying n support vectors.
    bias = nonlinsep_labels[index[0]]
    x1_sqaure_term = 0
    x2_sqaure_term = 0
    x1x2_term = 0
    for i in index:
        points_support_vectors.append(nonlinsep_points[i])
        labels_support_vectors.append(nonlinsep_labels[i])
        bias -= alpha[i] * nonlinsep_labels[i] * polynomial_kernel(nonlinsep_points[i], nonlinsep_points[index[0]].T)
        x1_sqaure_term += alpha[i] * nonlinsep_labels[i] * nonlinsep_points[i][0] * nonlinsep_points[i][0]
        x2_sqaure_term += alpha[i] * nonlinsep_labels[i] * nonlinsep_points[i][1] * nonlinsep_points[i][1]
        x1x2_term += alpha[i] * nonlinsep_labels[i] * 2 * nonlinsep_points[i][0] * nonlinsep_points[i][1]


    print('Polynomial kernel function:', '(X1.X2)**2')
    print('Support vectors indices:', index)
    print('Support lagrangian multiplier:', alpha[index])
    support_vectors = np.append(np.array(points_support_vectors), np.array(labels_support_vectors).reshape(-1, 1),
                                axis=1)
    print('Support Vectors:\n', support_vectors)
    print('bias:', bias)
    equation_line = str(round(x1_sqaure_term, 2)) + "x**2 " + ["", "+"][x2_sqaure_term >= 0] + str(round(x2_sqaure_term, 2)) + \
                    "y**2 " + ["", "+"][x1x2_term >= 0] + str(round(x1x2_term, 2)) + "xy " + \
                    ["", "+"][bias >= 0] + str(round(bias, 2))
    print('Equation of line:', equation_line, '= 0')

    # Plot the points and the hyperline/hyperplane
    plot_nonlinsep_points(nonlinsep_points, nonlinsep_labels, alpha, index, bias, np.array(points_support_vectors))

    y_predict = np.zeros(len(nonlinsep_labels))
    misclassified = 0
    for j in range(len(nonlinsep_points)):
        s = 0
        for i in index:
            s += alpha[i] * nonlinsep_labels[i] * polynomial_kernel(nonlinsep_points[i], nonlinsep_points[j].T)
        y_predict[j] = np.sign(s + bias)
        if y_predict[j] != nonlinsep_labels[j]:
            misclassified += 1
    print('Accuracy:', (N - misclassified) * 100/N, '%')
    print('=============================================\n\n')

    outfile.write('Non-linearly Separable Case:\n')
    outfile.write('Polynomial kernel function: ' + '(X1.X2)**2\n')
    outfile.write('Support vectors indices: ' + str(index) + '\n')
    outfile.write('Support lagrangian multiplier: ' + str(alpha[index]) + '\n')
    outfile.write('Support Vectors:\n' + str(support_vectors) + '\n')
    outfile.write('bias: ' + str(bias) + '\n')
    outfile.write('Equation of line: ' + str(equation_line) + '= 0\n')
    outfile.write('Accuracy: ' + str((N - misclassified) * 100/N) + '%\n')
    outfile.write('=============================================\n\n')


def sklearn_svm_linsep(X, y):

    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix

    svc = SVC(kernel='linear', C=1000)
    svc.fit(X, y)

    svc_weights = svc.coef_[0]
    svc_bias = svc.intercept_[0]
    print('sklearn-linearly separable case:')
    print('sklearn-weights:', svc_weights)
    print('sklearn-bias:', svc_bias)
    print('sklearn-support-vectors:\n', svc.support_vectors_)
    print('sklearn-support-indices:', svc.support_)
    print('sklear-confusion-matrix:\n', confusion_matrix(y, svc.predict(X)))
    print('sklearn-classification-report:\n', classification_report(y, svc.predict(X)))

    # plot_linsep_points(X, y, svc_weights, svc_bias, svc.support_vectors_)
    print('=============================================\n\n')


def sklearn_svm_nonlinsep(X,Y):

    from sklearn.svm import SVC


    print('sklearn-non-linearly separable case:')
    fignum = 1
    for kernel in ('linear', 'poly', 'rbf'):
        clf = SVC(kernel=kernel, degree=2, C=10)
        clf.fit(X, Y)

        # plot the line, the points, and the nearest vectors to the plane
        fig = plt.figure(fignum)
        plt.clf()

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                    edgecolors='k')

        XX, YY = np.meshgrid(np.linspace(-25, 25, 100), np.linspace(-25, 25, 100))
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        # plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])

        title = 'sklearn-' + kernel + ' kernel function'
        plt.title(title)
        plt.xlabel('X Label')
        plt.ylabel('Y Label')
        plt.legend(('+1', '-1'), loc='upper right')
        fignum = fignum + 1
        print(kernel, 'kernel score:', clf.score(X, Y))
        fig.savefig(title)
    plt.show()

    print('=============================================\n\n')


def main():

    linsep_df = pd.read_csv('linsep', sep=',', header=None)
    nonlinsep_df = pd.read_csv('nonlinsep', sep=',', header=None)
    linsep_array = linsep_df.to_numpy()
    nonlinsep_array = nonlinsep_df.to_numpy()

    linsep_points = linsep_array[:,:-1]
    linsep_labels = linsep_array[:,-1]
    nonlinsep_points = nonlinsep_array[:, :-1]
    nonlinsep_labels = nonlinsep_array[:, -1]

    output = open('output.txt', 'w')

    svm_linsep(linsep_points, linsep_labels, output)
    svm_nonlinsep(nonlinsep_points, nonlinsep_labels, output)

    output.close()

    sklearn_svm_linsep(linsep_points, linsep_labels)
    sklearn_svm_nonlinsep(nonlinsep_points, nonlinsep_labels)


if __name__ == '__main__':
    main()