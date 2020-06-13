#   Kushal Sharma
#   INF 552 Summer 2020
#   Homework 2 - K-means and GMM(EM)

import time
import matplotlib.pyplot as plt
import scipy.stats as st

import numpy as np
import pandas as pd
from random import seed
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Given the points, centroid value and no. of centroid, it can find
# the assignment of every point to each cluster. It does so by
# calculating distance from a point to every cluster and assigning
# to cluser having minimum distance.
def kmeans_assignment(points, centroid, k):
    assign = {}
    for i in range(k):
        assign[i] = []
    distance = []
    for i in range(len(points)):
        dist = []
        for j in range(len(centroid)):
            dist.append(np.sum(np.square(points[i] - centroid[j])))
        centroid_index = dist.index(min(dist))
        assign[centroid_index].append(points[i])
        distance.append(min(dist))
    for i in range(k):
        assign[i] = np.array(assign[i])
    return assign, np.sum(np.array(distance))


# Given the assignment and no. of clusters, it can calculate the value of
# new centroid
def kmeans_calc_centroid(assignment, k, d):
    new_centroid = np.zeros((k, d))
    for i in range(k):
        if (len(assignment[i]) > 0):
            new_centroid[i] = [(assignment[i].sum(axis=0))[0] / len(assignment[i]),
                               (assignment[i].sum(axis=0))[1] / len(assignment[i])]
    return new_centroid


# Main function to derive K-Means algorithm where convergence criteria is given by
# distance from a point to centroid of its nearest cluster assigned.
def kmeans(points, k, epochs):
    max_range = np.amax(points)
    # min_range = min(points.min(axis=0))
    min_range = np.amin(points)
    print("Max_Min:", max_range, min_range)
    n = points.shape[0]
    d = points.shape[1]
    # points = df.to_numpy()
    results = {}
    for trials in range(epochs):
        print("Epochs:", trials)
        centroid = np.zeros((k, d))
        new_centroid = np.random.uniform(-10, 10, size=(k, d))
        print("Initial Centroid:", new_centroid)
        iter = 0
        centroid_distance = []
        while(np.array_equal(new_centroid, centroid) == False):
            # if iter != 0:
            centroid = np.array(new_centroid)
            iter += 1
            assignment, distance = kmeans_assignment(points, new_centroid, k)
            new_centroid = kmeans_calc_centroid(assignment, k, d)
            centroid_distance.append(distance)
        print("Total iterations:", iter)
        print("Centroid:", new_centroid)
        print("Sum Centroid Distance:", distance)
        # plot_clusters(new_centroid, assignment)
        # plot_value_iteration(centroid_distance, iter, 'Sum Centroid Distance')
        results[distance] = [new_centroid, assignment, distance]
    print("Total unique distance:", results.keys())
    min_key = min(results.keys())
    plot_value_iteration(centroid_distance, iter, 'Sum Centroid Distance')
    return results[min_key]


# Plot the scatter points in assigned clusters with different colors
def plot_clusters(centroid, assignment):
    if (len(assignment[0]) > 0):
        plt.scatter(assignment[0][:,0], assignment[0][:,1], c='red', label='Cluster 1')
    if (len(assignment[1]) > 0):
        plt.scatter(assignment[1][:,0], assignment[1][:,1], c='blue', label='Cluster 2')
    if (len(assignment[2]) > 0):
        plt.scatter(assignment[2][:,0], assignment[2][:,1], c='green', label='Cluster 3')
    plt.scatter(centroid[:, 0], centroid[:, 1], s=100, c='magenta', label='Centroids')
    plt.title('Kmeans clusters')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    time.sleep(.2)


# Plot the scatter points and the gaussian curves of all the clusters.
def plot_gaussian(mean, cov, points):
    reg_cov = 1e-6 * np.identity(len(points[0]))
    x, y = np.meshgrid(np.sort(points[:, 0]), np.sort(points[:, 1]))
    XY = np.array([x.flatten(), y.flatten()]).T
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)
    ax0.scatter(points[:, 0], points[:, 1])
    ax0.set_title('GMM clusters')
    color = ['red', 'green', 'blue']
    i = 0
    for m, c in zip(mean, cov):
        c += reg_cov
        multi_normal = st.multivariate_normal(mean=m, cov=c)
        ax0.contour(np.sort(points[:, 0]), np.sort(points[:, 1]),
                    multi_normal.pdf(XY).reshape(len(points), len(points)), colors='black', alpha=0.3)
        ax0.scatter(m[0], m[1], c=color[i], zorder=10, s=100)
        i += 1
    plt.axis('equal')
    plt.show()
    time.sleep(.2)


# Plots the value of objective function alongwith the no. of iterations.
def plot_value_iteration(value, iteration, name):

    fig, ax = plt.subplots()
    ax.plot(range(0, iteration, 1), value)
    ax.set(xlabel='iterations', ylabel=name,
           title='Objective function value')
    plt.show()


# Given weights, mean and covariance, it can calculate the gaussian value of every cluser
# for every data point. That value is further normalized to calculate the probability
# distribution of each data point on every cluster.
def GMM_Expectation(points, k, weights, mean, covariance):
    # points = df.to_numpy()
    n = points.shape[0]
    d = points.shape[1]
    # E step:
    gaussian = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            mean_diff = points[i] - mean[j]
            inverse_covariance = np.linalg.inv(covariance[j])
            # print(inverse_covariance)
            exp_term = np.exp(-0.5 * np.dot(np.dot(mean_diff, inverse_covariance), mean_diff.T))
            # with np.errstate(invalid="ignore"):
            another_term = 1 / np.sqrt(np.power(2*np.pi, d) * np.linalg.det(covariance[j]))
            whole_term = weights[j] * another_term * exp_term
            gaussian[i][j] = whole_term
    #     for j in range(k):
    #         new_prob[i][j] = gaussian[i][j] / np.sum(gaussian[i])
    #     assign_index = np.argmax(new_prob[i])
    #     assign[assign_index].append(points[i])
    # for i in range(k):
    #     assign[i] = np.array(assign[i])
    # plot_gaussian(mean, covariance, points)

    return gaussian


# Given the gaussian value of every data point under every cluster, it can calculate
# weight, mean and covariance of every gaussian cluster.
def GMM_Maximization(points, k, gaussian):
    # M step:
    n = points.shape[0]
    d = points.shape[1]
    # points = df.to_numpy()
    new_prob = np.divide(gaussian, np.sum(gaussian, axis=1)[:, None])
    sum_prob = new_prob.sum(axis=0)
    # weights = np.ones(k)
    # mean = np.ones((k, d))
    # identity = np.identity(2)
    # covariance = np.array([identity, identity, identity])
    weights = []
    mean = []
    covariance = []
    new_covariance = []
    # new_cov = 0
    for i in range(k):
        weights.append(sum_prob[i] / sum(sum_prob))
        mean.append(np.sum(new_prob[:, i, None] * points, axis=0) / sum_prob[i])
        # for j in range(n):
        #     print(i, j, new_prob[j][i], points[j], mean[i])
        #     print((points[j] - mean[i]))
        #     print((points[j] - mean[i]).T)
        #     print(np.dot((points[j] - mean[i]).T, (points[j] - mean[i])))
        #     new_cov = new_prob[j][i] * np.dot((points[j] - mean[i]).T, (points[j] - mean[i]))
        #     print(i, j, new_cov)
        # new_covariance.append(new_cov / sum_prob[i])
        mean_diff = points - mean[i]
        weighted_mean_diff = np.multiply(new_prob[:, i, None], mean_diff)
        covariance.append(np.dot(weighted_mean_diff.T, mean_diff) / sum_prob[i])
    # print("Old covariance:", covariance)
    return weights, mean, covariance


# Main function to run GMM algorithm in iteration using Expectation-Maximization
# method. It can either take the centroid value from K-Means or take the initialized
# value of weights, mean and covariance. Log likelihood of the gaussian value is considered
# as objective function to converge. Only upto 3 decimal point value is considered
# for convergence as no significant imporvement is observed beyond that.
def GMM(points, k, epochs, centroid, assignment):
    # Assume that each datapoint has equal distribution on every cluster
    # r = 1/k
    d = points.shape[1]
    n = points.shape[0]
    # points = df.to_numpy()
    results = {}
    for trials in range(epochs):
        print("Epochs:", trials)
        weights = []
        mean = []
        covariance = []
        # weights = np.ones(k) / k
        # mean = np.ones((k, d))
        # identity = np.identity(d)
        # covariance = np.array([identity, identity, identity])
        if np.array_equal(centroid, np.ones((k, d))) == True:
            # mean = np.ones((k, d))
            # mean = np.random.uniform(-10, 10, size=(k, d))
            gaussian = np.random.uniform(0, 10000, size=(n, k))
            weights, mean, covariance = GMM_Maximization(points, k, gaussian)
        else:
            mean = centroid
            for i in range(k):
                weights.append(len(assignment[i][:, 0]) / n)
                covariance.append(np.dot((points - mean[i]).T, (points - mean[i])) / len(points))
        print("Initial Weights:", weights)
        print("Initial Mean:", mean)
        print("Initial Covariance:", covariance)
        iter = 0
        log_likelihood = 0
        new_log_likelihood = 1
        log_likelihood_estimation = []
        while (np.array_equal(log_likelihood, new_log_likelihood) == False):
            log_likelihood = new_log_likelihood
            iter += 1
            # E step:
            gaussian = GMM_Expectation(points, k, weights, mean, covariance)
            # M step:
            weights, mean, covariance = GMM_Maximization(points, k, gaussian)
            new_log_likelihood = round(np.sum(np.log(np.sum(gaussian, axis=1))), 3)
            # print("Log prob:", new_log_likelihood)
            log_likelihood_estimation.append(new_log_likelihood)
        print("Total Iterations:", iter)
        print("Weights:", weights)
        print("Mean:", mean)
        print("Covariance:", covariance)
        print("Log Likelihood:", log_likelihood)
        # plot_gaussian(mean, covariance, points)
        # plot_value_iteration(log_likelihood_estimation, iter, 'Log Likelihood')
        results[log_likelihood] = [weights, mean, covariance, log_likelihood]
    print("Total unique log likelihood:", results.keys())
    max_key = max(results.keys())
    plot_value_iteration(log_likelihood_estimation, iter, 'Log Likelihood')
    return results[max_key]


# K-Means algorithm using sklearn library
def sklearn_kmeans(points, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(points)
    new_centroid = kmeans.cluster_centers_
    assignment, distance = kmeans_assignment(points, new_centroid, k)
    return new_centroid, assignment, distance


# GMM algorithm using sklearn library.
def sklearn_gmm(points, k):
    gmm = GaussianMixture(n_components=k)
    gmm.fit(points)
    weights = gmm.weights_
    mean = gmm.means_
    covariance = gmm.covariances_
    log_likelihood = gmm.lower_bound_
    return weights, mean, covariance, log_likelihood


def main(K):
    dataFrame = pd.read_csv('clusters.txt', sep=",", header=None)
    points = dataFrame.to_numpy()
    D = points.shape[1]

    f = open("output.txt", "w")

    kmeans_epochs = 20
    centroid, assignment, distance = kmeans(points, K, kmeans_epochs)
    print("K-means:")
    print("Best results from kmeans after", kmeans_epochs, "epochs:")
    print("Best centroid value:", centroid)
    print("Best closest centroid distance:", distance)
    plot_clusters(centroid, assignment)
    f.write("K-means:\n")
    f.write("Best results from kmeans after " + str(kmeans_epochs) + " epochs:\n")
    f.write("Best centroid value: " + str(centroid) + '\n')
    f.write("Best closest centroid distance: " + str(distance) + '\n')
    f.write('\n' + '\n')

    GMM_epochs = 1
    weights, mean, covariance, log_likelihood = GMM(points, K, GMM_epochs, centroid, assignment)
    print("GMM with EM:")
    print("With passing centroid value from Kmeans to GMM")
    print("Best results from GMM after", GMM_epochs, "epochs:")
    print("Best Weights:", weights)
    print("Best Mean:", mean)
    print("Best Covariance:", covariance)
    print("Best log likelihood:", log_likelihood)
    plot_gaussian(mean, covariance, points)
    f.write("GMM with EM:\n")
    f.write("With passing centroid value from Kmeans to GMM\n")
    f.write("Best results from GMM after " + str(GMM_epochs) + " epochs:\n")
    f.write("Best Weights: " + str(weights) + '\n')
    f.write("Best Mean: " + str(mean) + '\n')
    f.write("Best Covariance: " + str(covariance) + '\n')
    f.write("Best log likelihood: " + str(log_likelihood) + '\n')
    f.write('\n' + '\n')

    GMM_epochs = 20
    assignment = 1
    centroid = np.ones((K, D))
    weights, mean, covariance, log_likelihood = GMM(points, K, GMM_epochs, centroid, assignment)
    print("Without passing centroid value from Kmeans to GMM")
    print("Best results from GMM after", GMM_epochs, "epochs:")
    print("Best Weights:", weights)
    print("Best Mean:", mean)
    print("Best Covariance:", covariance)
    print("Best log likelihood:", log_likelihood)
    plot_gaussian(mean, covariance, points)
    f.write("Without passing centroid value from Kmeans to GMM\n")
    f.write("Best results from GMM after " + str(GMM_epochs) + " epochs:\n")
    f.write("Best Weights: " + str(weights) + '\n')
    f.write("Best Mean: " + str(mean) + '\n')
    f.write("Best Covariance: " + str(covariance) + '\n')
    f.write("Best log likelihood: " + str(log_likelihood) + '\n')
    f.write('\n' + '\n')


    new_centroid, assignment, distance = sklearn_kmeans(points, K)
    print("Sklearn_Kmeans:")
    print("Centroid:", new_centroid)
    print("distance:", distance)
    plot_clusters(new_centroid, assignment)
    f.write("Sklearn_Kmeans:\n")
    f.write("Centroid: " + str(new_centroid) + '\n')
    f.write("distance: " + str(distance) + '\n')
    f.write('\n' + '\n')

    weights, mean, covariance, log_likelihood = sklearn_gmm(points, K)
    print("Sklearn_GMM:")
    print("Weights:", weights)
    print("Mean:", mean)
    print("Covariance:", covariance)
    print("Log Likelihood:", log_likelihood)
    plot_gaussian(mean, covariance, points)
    f.write("Sklearn_GMM:\n")
    f.write("Weights: " + str(weights) + '\n')
    f.write("Mean: " + str(mean) + '\n')
    f.write("Covariance: " + str(covariance) + '\n')
    f.write("Log Likelihood: " + str(log_likelihood) + '\n')
    f.write('\n' + '\n')

    f.close()


if __name__ == '__main__':
    # np.set_printoptions(suppress=True, formatter={'all':lambda x: str(x)})
    num_clusters = 3
    main(num_clusters)