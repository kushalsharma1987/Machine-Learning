# Kushal Sharma
# kushals@usc.edu
# HW 3 - PCA and FastMap
# INF 552 Summer 2020

import math
import random

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def PCA_sklearn(points, N, n, k):
    from sklearn.decomposition import PCA
    points -= np.mean(points, axis=0)

    cov_mat = np.cov(points.T)
    pca = PCA(n_components=k)
    X = pca.fit_transform(points)
    print("PCA_Sklearn_points:\n", X)

    eigenvecmat = []
    for eigenvector in pca.components_:
        if len(eigenvecmat) == 0:
            eigenvecmat = eigenvector
        else:
            eigenvecmat = np.vstack((eigenvecmat, eigenvector))
    print("eigenvector-matrix(sklearn):\n", eigenvecmat)
    eigenvalue = pca.explained_variance_
    print(eigenvalue)
    plot_points_eigenvector(X, eigenvecmat.T, eigenvalue, filename='PCA_sklearn_plot')


def PCA_SVD(points, N, n, k):
    points -= np.mean(points, axis=0)
    # print(points)

    from numpy.linalg import svd
    U, S, Vt = svd(points, full_matrices=False, compute_uv=True)
    # print("U:\n", U)
    # print("S:\n", S)
    # print("Vt:\n", Vt)

    Sigma = np.zeros((6000, 3), dtype=float)
    Sigma[:3, :3] = np.diag(S)
    # print("Sigma:\n", Sigma)

    # points_svd = np.dot(U, np.dot(Sigma, Vt))
    points_svd = np.dot(U, np.diag(S))
    print("PCA_SVD_points:\n", points_svd)


def FASTMAP():
    num_reduced_dimensions = 2
    k = num_reduced_dimensions
    fastmap_df = pd.read_csv('fast-map-data.txt', sep='\t', header=None)
    fastmap_distance_array = fastmap_df.to_numpy()
    num_unique_objects = np.unique(fastmap_distance_array[:, :2]).tolist()
    coords = {k:[] for k in num_unique_objects}
    for i in range(k):
        object_list = fastmap_distance_array[:, :2].tolist()
        farthest_pair = fastmap_farthest_pair(fastmap_distance_array, num_unique_objects, object_list)
        print(farthest_pair)
        coords = fastmap_find_coordinate(fastmap_distance_array, farthest_pair, coords, num_unique_objects, object_list)
        fastmap_distance_array = fastmap_update_distance(fastmap_distance_array, coords, i)
        # print(fastmap_distance_array)
    points = np.array(list(coords.values()))
    print("Fastmap objects embedded points:\n", points)
    fastmap_wordlist = pd.read_csv('fastmap-wordlist.txt', header=None).values.flatten().tolist()
    print("Wordlist:\n", fastmap_wordlist)
    plot_points_object(points, annotation=fastmap_wordlist, filename='Fastmap_plot')


def fastmap_update_distance(distance_matrix, coords, coords_index):
    new_distance_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1]))
    for i in range(len(distance_matrix)):
        orig_distance = distance_matrix[i, 2]
        a = distance_matrix[i, 0]
        b = distance_matrix[i, 1]
        coords_distance = coords[a][coords_index] - coords[b][coords_index]
        new_distance = math.sqrt(abs(math.pow(orig_distance, 2) - math.pow(coords_distance, 2)))
        new_distance_matrix[i, 0] = int(a)
        new_distance_matrix[i, 1] = int(b)
        new_distance_matrix[i, 2] = float(new_distance)
    return new_distance_matrix


def fastmap_find_coordinate(distance_matrix, farthest_pair, coords, num_unique_objects, object_list):
    a = farthest_pair[0]
    b = farthest_pair[1]
    ind_a_b = fastmap_get_index(object_list, a, b)
    dist_a_b = distance_matrix[ind_a_b, 2]
    for i in num_unique_objects:
        ind_a_i = fastmap_get_index(object_list, i, a)
        if ind_a_i == -1:
            dist_a_i = 0
        else:
            dist_a_i = distance_matrix[ind_a_i, 2]
        ind_i_b = fastmap_get_index(object_list, i, b)
        if ind_i_b == -1:
            dist_i_b = 0
        else:
            dist_i_b = distance_matrix[ind_i_b, 2]
        coord_i = (math.pow(dist_a_i, 2) + math.pow(dist_a_b, 2) - math.pow(dist_i_b, 2)) / (2 * dist_a_b)
        coords[i].append(coord_i)
        # print(coords)
    return coords


def fastmap_farthest_pair(distance_matrix, num_unique_objects, object_list):
    random_object = random.choice(num_unique_objects)
    dist = {}
    pivot_list = []
    new_pivot = random_object
    while new_pivot not in pivot_list:
        pivot_list.append(new_pivot)
        for i in num_unique_objects:
            index = fastmap_get_index(object_list, i, new_pivot)
            if index == -1:
                dist[0] = [i, new_pivot]
            else:
                distance = distance_matrix[index, 2]
                dist[distance] = [i, new_pivot]
        max_dist = max(dist.keys())
        old_pivot = new_pivot
        new_pivot = dist[max_dist][0]
    print("Farthest pair:", old_pivot, 'and', new_pivot, " with distance=", max_dist)
    return [old_pivot, new_pivot]


def fastmap_get_index(object_list, x, y):
    if [x, y] in object_list:
        return object_list.index([x, y])
    elif [y, x] in object_list:
        return object_list.index([y, x])
    else:
        return -1


def PCA(points, N, n, k):
    mean_3d = np.divide(np.sum(points, axis=0), N)
    mean_diff_3d = points - mean_3d
    covariance_3d = np.divide(np.dot(mean_diff_3d.T, mean_diff_3d), N)
    print("Covariance Matrix(3d points):\n", covariance_3d)
    eigenvalue, eigenvector = np.linalg.eig(covariance_3d)
    print("Eigenvector(3d points):\n", eigenvector)
    print("Eigenvalue(3d points):\n", eigenvalue)
    plot_points_eigenvector(points, eigenvector, eigenvalue, filename='PCA_plot_3d')


    max_arg = eigenvalue.argsort()[::-1][:k]
    eigenvector_truncated = eigenvector[:, max_arg]
    eigenvalue_truncated = eigenvalue[max_arg]
    print("Eigenvector_truncated:\n", eigenvector_truncated)
    print("Eigenvalue_truncated:\n", eigenvalue_truncated)

    with open("PCA_direction.txt", "w") as file:
        file.write("PCA direction of first two component:\n")
        for i in range(k):
            file.write("Direction " + str(i+1) + ": " + str(eigenvector_truncated[:, i]) + '\n')
        file.write("\n\nPCA magnitude of first two component:\n")
        file.write(str(eigenvalue_truncated))
    points_2d = []

    for i in range(N):
        points_2d.append(np.dot(points[i], eigenvector_truncated))
    points_2d = np.array(points_2d)
    mean_2d = np.divide(np.sum(points_2d, axis=0), N)
    mean_diff_2d = points_2d - mean_2d
    covariance_2d = np.divide(np.dot(mean_diff_2d.T, mean_diff_2d), N)
    print("Covariance Matrix(2d points):\n", covariance_2d)
    eigenvalue_2d , eigenvector_2d = np.linalg.eig(covariance_2d)
    print("Eigenvector(2d points):\n", eigenvector_2d)
    print("Eigenvvalue(2d points):\n", eigenvalue_2d)
    print("PCA_EigenDecomposition_points(2d points):\n", points_2d)
    # plot_points_eigenvector(points_2d, eigenvector=eigenvector_truncated, eigenvalue=eigenvalue_truncated, filename='PCA_plot_2d')
    plot_points_eigenvector(points_2d, eigenvector_truncated, eigenvalue_truncated, filename='PCA_plot_2d')


def plot_points_object(points, annotation=False, filename='plot'):
    x = points[:, 0]
    y = points[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c='black', marker='o')
    if annotation != False:
        k = 0
        for i, j in zip(x, y):
            ax.annotate(annotation[k], xy=(i-0.2, j + 0.2))
            k += 1
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    plt.title(filename)
    plt.show()
    fig.savefig(filename + '.png', bbox_inches='tight')


def plot_points_eigenvector(points, eigenvector, eigenvalue, filename='plot'):
    fig = plt.figure()
    if points.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    xs = points[:, 0]
    ys = points[:, 1]
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    if points.shape[1] == 3:
        zs = points[:, 2]
        mean_z = np.mean(zs)
        ax.plot(xs, ys, zs, 'o', markersize=10, color='black', alpha=0.2)
        ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='orange', alpha=0.5)
    else:
        ax.plot(xs, ys, 'o', markersize=10, color='black', alpha=0.2)
        ax.plot([mean_x], [mean_y], 'o', markersize=10, color='orange', alpha=0.5)
    c = ['red','yellow','cyan']
    if eigenvector.size != 0:
        i =0
        for v in eigenvector.T:
            print("Vector", i, ":", v)
            v = v * eigenvalue[i]
            if points.shape[1] == 3:
                ax.plot([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], color=c[i], alpha=.8, lw=3)
            else:
                # mean_z = np.zeros(1)
                # mean = np.array([mean_x, mean_y, mean_z])
                # eig_vec1 = eigenvector[:, 0]
                # eig_vec2 = eigenvector[:, 1]
                # plt.quiver(*mean, *eig_vec1, color=['red'], scale=2)
                # plt.quiver(*mean, *eig_vec2, color=['yellow'], scale=2)

                ax.plot([mean_x, v[0]], [mean_y, v[1]], color=c[i], alpha=.8, lw=3)
            i+=1

    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    if points.shape[1] > 2:
        ax.set_zlabel('z_values')

    plt.title(filename)

    plt.draw()
    plt.show()
    fig.savefig(filename + '.png', bbox_inches='tight')


def main():
    num_reduced_dimensions = 2
    k = num_reduced_dimensions
    pca_df = pd.read_csv('pca_data.txt', sep='\t', header=None)
    N = pca_df.shape[0]
    n = pca_df.shape[1]
    points = pca_df.to_numpy()
    # points -= np.mean(points, axis=0)

    PCA(points, N, n, k)
    PCA_SVD(points, N, n, k)
    PCA_sklearn(points, N, n, k)
    FASTMAP()


if __name__ == '__main__':
    main()