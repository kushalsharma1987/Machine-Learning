import time
import matplotlib.pyplot as plt
import scipy.stats as st

import numpy as np
import pandas as pd
from random import seed
from random import random


def kmeans_assignment(points, centroid, k):
    assign = {}
    for i in range(k):
        assign[i] = []
    # assign[0] = []
    # assign[1] = []
    # assign[2] = []
    # print(assign)
    Q = []
    for i in range(len(points)):
        dist = []
        for j in range(len(centroid)):
            # print(points[i], centroid[j])
            # print(np.sum(np.square(points[i] - centroid[j])))
            # print(np.linalg.norm(points[i] - centroid[j]))
            # dist[j] = np.linalg.norm(points[i] - centroid[j])
            dist.append(np.sum(np.square(points[i] - centroid[j])))
        # print(dist)
        centroid_index = dist.index(min(dist))
        # print(centroid_index)
        assign[centroid_index].append(points[i])
        Q.append(min(dist))
    for i in range(k):
        assign[i] = np.array(assign[i])
    return assign, np.sum(np.array(Q))


def kmeans_calc_centroid(assignment, k, d):
    new_centroid = np.zeros((k, d))
    for i in range(k):
        if (len(assignment[i]) > 0):
            new_centroid[i] = [(assignment[i].sum(axis=0))[0] / len(assignment[i]),
                               (assignment[i].sum(axis=0))[1] / len(assignment[i])]
    return new_centroid


def kmeans(df, k):
    max_range = max(df.max(axis=0))
    min_range = min(df.min(axis=0))
    print("Max_Min:", max_range, min_range)
    # seed(1)
    # centroid = np.random.uniform(-10, 10, size=(3, 2))
    # print("Initial centroid:", centroid)
    n = df.shape[0]
    d = df.shape[1]
    # print(df.shape[1])
    points = df.to_numpy()
    # print("Sum of data points column:", points.sum(axis=0))
    # print(points)
    results = {}
    for trials in range(1):
        seed(trials)
        centroid = np.random.uniform(-10, 10, size=(k, d))
        # print("Initial centroid:", centroid)
        new_centroid = np.zeros((k, d))
        iter = 0
        while(np.array_equal(new_centroid, centroid) == False):
            if iter != 0:
                centroid = np.array(new_centroid)
            iter += 1
            assignment, Q = kmeans_assignment(points, centroid, k)
            # print("Distance:", Q)
            # pprint(assign)
            # print((assign[0].sum(axis=0)), (assign[0].sum(axis=0))[0], (assign[0].sum(axis=0))[1])
            # print(len(assign[0]))
            new_centroid = kmeans_calc_centroid(assignment, k, d)
            # centroid[0] = [(assign[0].sum(axis=0))[0]/len(assign[0]), (assign[0].sum(axis=0))[1]/len(assign[0])]
            # centroid[0] = [(assign[0].sum(axis=0))[0] / len(assign[0]), (assign[0].sum(axis=0))[1] / len(assign[0])]
            # centroid[0] = [(assign[0].sum(axis=0))[0] / len(assign[0]), (assign[0].sum(axis=0))[1] / len(assign[0])]
            # print("Iteration:", iter, new_centroid)
            plot_clusters(new_centroid, assignment)
        results[Q] = [new_centroid, assignment]
        # time.sleep(1)
        # plot_clusters(new_centroid, assignment)
    min_key = min(results.keys())
    # print(results, len(results))
    # print("Optimal solution:", results[min_key], min_key)
    return results[min_key][1]


def plot_clusters(centroid, assignment):
    if (len(assignment[0]) > 0):
        plt.scatter(assignment[0][:,0], assignment[0][:,1], c='red', label='Cluster 1')
    if (len(assignment[1]) > 0):
        plt.scatter(assignment[1][:,0], assignment[1][:,1], c='blue', label='Cluster 2')
    if (len(assignment[2]) > 0):
        plt.scatter(assignment[2][:,0], assignment[2][:,1], c='green', label='Cluster 3')
    plt.scatter(centroid[:, 0], centroid[:, 1], s=100, c='magenta', label='Centroids')
    plt.title('Clusters of points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    time.sleep(.2)


def plot_gaussian(mean, cov, points):
    # if (len(assignment[0]) > 0):
    #     plt.scatter(assignment[0][:,0], assignment[0][:,1], c='red', label='Cluster 1')
    # if (len(assignment[1]) > 0):
    #     plt.scatter(assignment[1][:,0], assignment[1][:,1], c='blue', label='Cluster 2')
    # if (len(assignment[2]) > 0):
    #     plt.scatter(assignment[2][:,0], assignment[2][:,1], c='green', label='Cluster 3')
    # plt.scatter(centroid[:, 0], centroid[:, 1], s=100, c='magenta', label='Centroids')
    # plt.contourf(assignment[0][:,0], assignment[0][:,1], gaussian[0], cmap='Reds')
    # plt.contourf(assignment[1][:, 0], assignment[1][:, 1], gaussian[1], cmap='Blues')
    # plt.contourf(assignment[2][:, 0], assignment[2][:, 1], gaussian[2], cmap='Greens')
    # plt.colorbar()
    # plt.show()
    # plt.title('Clusters of points')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.plot(assignment[0][:, 0], assignment[0][:,1], gaussian[0])
    # plt.plot(assignment[1][:, 0], assignment[1][:, 1], gaussian[1])
    # plt.plot(assignment[2][:, 0], assignment[2][:, 1], gaussian[2])
    # plt.show()
    k = len(mean)
    # Extract x and y
    # for i in range(k):
    #     x = assignment[i][:, 0]
    #     y = assignment[i][:, 1]  # Define the borders
    #     deltaX = (max(x) - min(x)) / 10
    #     deltaY = (max(y) - min(y)) / 10
    #     xmin = min(x) - deltaX
    #     xmax = max(x) + deltaX
    #     ymin = min(y) - deltaY
    #     ymax = max(y) + deltaY
    #     # print(xmin, xmax, ymin, ymax)  # Create meshgrid
    #     xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
    reg_cov = 1e-6 * np.identity(len(points[0]))
    x, y = np.meshgrid(np.sort(points[:, 0]), np.sort(points[:, 1]))
    XY = np.array([x.flatten(), y.flatten()]).T
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)
    ax0.scatter(points[:, 0], points[:, 1])
    ax0.set_title('Initial state')
    for m, c in zip(mean, cov):
        c += reg_cov
        multi_normal = st.multivariate_normal(mean=m, cov=c)
        ax0.contour(np.sort(points[:, 0]), np.sort(points[:, 1]),
                    multi_normal.pdf(XY).reshape(len(points), len(points)), colors='black', alpha=0.3)
        ax0.scatter(m[0], m[1], c='grey', zorder=10, s=100)
            # plt.plot(assignment[i][:, 0], assignment[i][:, 1], 'x')
    # plt.axis('equal')
    plt.show()

    # for i in range(k):
    #     x, y = np.random.multivariate_normal(mean[i], cov[i], 50).T
    #     print(x, y)
    #     plt.plot(x, y, 'x')
    # plt.axis('equal')
    # plt.show()

    # fig = plt.figure(figsize=(13, 7))
    # ax = plt.axes(projection='3d')
    # w = ax.plot_wireframe(xx, yy, gaussian)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('PDF')
    # ax.set_title('Wireframe plot of Gaussian 2D KDE');
    time.sleep(.2)


def EM(df, k):
    # Assume that each datapoint has equal distribution on every cluster
    # r = 1/k
    d = df.shape[1]
    n = df.shape[0]
    points = df.to_numpy()
    weights = np.ones(k)
    # mean = np.ones((k, d))
    mean = np.random.uniform(-10, 10, size=(k, d))
    print("Initial Mean:", mean)
    covariance = np.ones((k, d, d))
    # covariance = np.identity(points.shape[1], dtype=np.float64)
    # assignment = kmeans(df, k)
    for i in range(k):
        # weights[i] = len(assignment[i]) / n
        # mean[i] = assignment[i].sum(axis=0) / len(assignment[i])
        # print(assignment[i] - mean[i])
        # print((assignment[i] - mean[i]).T)
        # covariance[i] = (np.dot((assignment[i] - mean[i]).T, (assignment[i] - mean[i]))) / len(assignment[i])
        covariance[i] = (np.dot((points - mean[i]).T, (points - mean[i]))) / len(points)
        # covariance[i] = np.identity(2, dtype=np.float128)
        # covariance[i] = (1 / n) * np.sum(np.square(assignment[i] - mean[i]))
    # print(weights)
    # print(mean)
    # print(covariance)
    # print(kmeans_assign)
    prob = 1/k * np.ones((n, k))
    new_prob = np.zeros((n, k))
    iter = 0
    log_likehood = 0
    new_log_likehood = 1
    prob_log = 0
    new_prob_log = 1
    log_likelihood_estimation = []
    # print(prob)
    # while(np.array_equal(prob, new_prob) == False):
    # while (np.array_equal(log_likehood, new_log_likehood) == False):
    while (np.array_equal(round(prob_log, 3), round(new_prob_log, 3)) == False):
        assign = {}
        for i in range(k):
            assign[i] = []
        if iter != 0:
            log_likehood = new_log_likehood
            # prob_log = np.sum(np.log(new_prob))
            prob_log = new_prob_log
        iter += 1
    # E step:
        gaussian = np.zeros((150, 3))
        for i in range(n):
            # gaussian = []
            for j in range(k):
                mean_diff = points[i] - mean[j]
                inverse_covariance = np.linalg.inv(covariance[j])
                # print(inverse_covariance)
                exp_term = np.exp(-0.5 * np.dot(np.dot(mean_diff, inverse_covariance), mean_diff.T))
                another_term = 1 / np.sqrt(np.power(2*np.pi, d) * np.linalg.det(covariance[j]))
                whole_term = weights[j] * another_term * exp_term
                gaussian[i][j] = whole_term
            for j in range(k):
                new_prob[i][j] = gaussian[i][j] / np.sum(gaussian[i])
            assign_index = np.argmax(new_prob[i])
            assign[assign_index].append(points[i])
        for i in range(k):
            assign[i] = np.array(assign[i])
        # print(assign)
        plot_gaussian(mean, covariance, points)
        # plot_clusters(mean, assign)
        # plot_gaussian(assignment, gaussian)
    # M step:
        sum_prob = new_prob.sum(axis=0)
        # print(sum_prob, sum(sum_prob))
        # prob_0 = new_prob[:, 0]
        # print(prob_0[:, None])
        # print(np.matmul(prob_0[:, None].T, points))
        # for i in range(k):
        #     mean[i] = np.sum(new_prob[:,i, None] * points, axis=0)
        # print("Mult mean:", mean)
        # mean = np.dot(new_prob.T, points)
        # print("Dot Mean", mean)
        # mean = np.matmul(new_prob.T, points)
        # print("Matmul Mean:", mean)

        # print(mean)
        # print(points[:5])
        for i in range(k):
            weights[i] = sum_prob[i] / sum(sum_prob)
            # for j in range(n):
                # mean[i] = np.multiply(new_prob[:, 0], points[j])
            # mean[i] = mean[i] / sum_prob[i]
            mean[i] = np.sum(new_prob[:, i, None] * points, axis=0) / sum_prob[i]
            # print(mean[i])
            mean_diff = points - mean[i]
            # print(mean_diff[:5])
            weighted_mean_diff = np.multiply(new_prob[:, i, None], mean_diff)
            # print(weighted_mean_diff)
            # # print(mean_diff, type(mean_diff))
            # # print(weighted_mean_diff, type(weighted_mean_diff))
            # # print(np.dot((points - mean[i]).T, (points - mean[i])))
            # covariance[i] = np.dot(mean_diff, weighted_mean_diff).sum(axis=0) / sum_prob[i]
            # covariance[i] = new_prob[:,i, None] * np.dot(mean_diff.T, mean_diff)
            # print("New Cov:", covariance[i])
            covariance[i] = np.dot(weighted_mean_diff.T, mean_diff) / sum_prob[i]
        print("EM Iteration:", iter)
        # print("Weights:", weights)
        print("Mean:", mean)
        # print("Covariance:", covariance)
        # new_prob_log = np.sum(np.log(new_prob))
        # new_prob_log = np.sum(np.log(sum_prob))
        new_prob_log = np.sum(np.log(np.sum(gaussian, axis=1)))
        # log_likelihood_estimation.append(np.log(np.sum([k * st.multivariate_normal(self.mu[i], self.cov[j]).pdf(X) for k, i, j in
        #                                       zip(self.pi, range(len(self.mu)), range(len(self.cov)))])))
        print("Log prob:", new_prob_log)
        log_likelihood_estimation.append(new_prob_log)
    plot_gaussian(mean, covariance, points)
    fig2 = plt.figure(figsize=(10, 10))
    ax1 = fig2.add_subplot(111)
    ax1.set_title('Log-Likelihood')
    ax1.plot(range(0, iter, 1), log_likelihood_estimation)
    plt.show()

def main(K):
    dataFrame = pd.read_csv('clusters.txt', sep=",", header=None)
    # print(dataFrame)
    # kmeans(dataFrame, K)
    EM(dataFrame, K)


if __name__ == '__main__':
    # np.set_printoptions(suppress=True, formatter={'all':lambda x: str(x)})
    num_clusters = 3
    main(num_clusters)