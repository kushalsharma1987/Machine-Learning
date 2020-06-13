# Name - Kushal Sharma
# Course - INF 552 Summer 2020
# Assignment - Homework 1
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing


def entropy(col):
    (unique, counts) = np.unique(col, return_counts=True)
    sum_total = np.sum(counts)
    each = np.zeros(len(unique))
    for i in range(len(unique)):
        each[i] = -(counts[i]/sum_total) * np.log2(counts[i]/sum_total)
        # print("Entropy:", counts[i], sum_total, each[i])
    return round(np.sum(each),5)


def infogain(df, col, target_col, base_entropy):
    # print(df.columns[0])
    # print(df)
    (vals, counts) = np.unique(df[col], return_counts=True)
    # print(vals, counts)
    sum_vals = np.sum(counts)
    each = np.zeros(len(vals))
    for i in range(len(vals)):
        # For each unique value in attr_column, check the entropy for target_column
        # print(df.loc[df[col] == vals[i]])
        each[i] = (counts[i]/sum_vals) * entropy(df.loc[df[col] == vals[i]][target_col].values)
        each[i] = round(each[i], 5)
        # print("infogain:", counts[i], sum_vals, each[i])
    attr_entropy = np.sum(each)
    attr_entropy = round(attr_entropy, 5)
    # print(col, attr_entropy)
    return round(base_entropy - attr_entropy, 5)


def decisiontree(dataFrame, target_col, max_class, node_class=None):

    base_entropy = entropy(dataFrame[target_col].values)
    # print("Base Entropy:", base_entropy)
    # if the target class is pure, meaning all data instances have same value, return that value.
    if base_entropy == 0:
        return np.unique(dataFrame[target_col])
    # if there are no more examples left, then return the majority value of target col from original data
    elif len(dataFrame) == 0:
        return max_class
    # if there are no more attributes other than target col, return the majority value from previous split attribute
    elif len(dataFrame.columns) == 1:
        return node_class
    # Grow the tree in other cases.
    else:
        (target_vals, target_counts) = np.unique(dataFrame[target_col], return_counts=True)
        node_class = target_vals[np.argmax(target_counts)]

        info_gain = {}
        for col in dataFrame.columns:
            # print(col, type(col))
            if col == target_col:
                continue
            info_gain[col] = infogain(dataFrame[[col, target_col]], col, target_col, base_entropy)
        print(info_gain)

        # Pick best attribute to be expanded further.
        itemMaxValue = max(info_gain.items(), key=lambda x: x[1])
        max_branch = {}
        # Iterate over all the items in dictionary to find keys with max value
        for key, value in info_gain.items():
            if value == itemMaxValue[1]:
                # Capture the no. of choices for each branch.
                (best_vals, best_counts) = np.unique(dataFrame[key], return_counts=True)
                max_branch[key] = len(best_vals)
        # Pick the attribute with minimum no. of choices which would result in shorter decision tree
        best_col = min(max_branch, key=max_branch.get)
        print(max_branch)
        print(best_col)
        tree = {best_col: {}}

        # Go through every choice of the best attribute and expand them depth first fashion through recursion.
        for val in np.unique(dataFrame[best_col]):
            sub_data = dataFrame.loc[dataFrame[best_col] == val].drop(best_col, 1)
            # print("Best_Col:", val)
            # print(sub_data)
            sub_tree = decisiontree(sub_data, target_col, max_class, node_class)
            tree[best_col][val] = sub_tree
        return tree


def predict(query, tree, default):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result, default)
            else:
                return result


# def traintestsplit(dataset):
#     # training_data = dataset.iloc[:round(.80 * len(dataset))].reset_index(drop=True)
#     # We drop the index respectively relabel the index
#     training_data = dataset.iloc[:22].reset_index(drop=True)
#     # testing_data = dataset.iloc[round(.80 * len(dataset)):].reset_index(drop=True)
#     testing_data = dataset.iloc[21:].reset_index(drop=True)
#     return training_data, testing_data
#
#
# def test(data, tree, target_col, default):
#     # Create new query instances by simply removing the target feature column from the original dataset and
#     # convert it to a dictionary
#     queries = data.iloc[:, :-1].to_dict(orient="records")
#
#     # Create a empty DataFrame in whose columns the prediction of the tree are stored
#     predicted = pd.DataFrame(columns=["predicted"])
#
#     # Calculate the prediction accuracy
#     for i in range(len(data)):
#         predicted.loc[i, "predicted"] = predict(queries[i], tree, default)
#     print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data[target_col]) / len(data)) * 100, '%')


def main():
    dataFrame = pd.read_csv('dt_data.txt', sep="[\s\d():,;\n]+", engine='python')
    dataFrame.dropna(axis='columns', inplace=True)
    dataFrame.rename(columns={'Beer':'Enjoy', 'Favorite': 'Favorite Beer'}, inplace=True)
    target_col = 'Enjoy'
    (vals, counts) = np.unique(dataFrame[target_col], return_counts=True)
    max_class = vals[np.argmax(counts)]
    dTree = decisiontree(dataFrame, target_col, max_class)
    pprint(dTree)

    query_text = pd.read_csv('query.txt', sep="; ", header=None, engine='python')
    query_arr = np.array(query_text.values.reshape(-1))
    # print(query_arr)
    query = {}
    for item in query_arr:
        key_value = item.split(' = ')
        # query[item[0]] = item[1]
        query[key_value[0]] = key_value[1]
    print(query)

    prediction = predict(query, dTree, max_class)
    print(prediction)


# ## SKLEARN Training and Testing
#     le = preprocessing.LabelEncoder()
#     dfData = dataFrame.values
#     for i in range(len(dataFrame.columns)):
#         dfData[:, i] = le.fit_transform(dfData[:, i])
#     print(dfData)
#     X = dfData[:,:-1]
#     y = dfData[:,-1]
#     y = y.astype('int')
#     print(X, type(X))
#     print(y, type(y))
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#     clf = tree.DecisionTreeClassifier(criterion="entropy")
#     # print(clf, type(clf))
#     model = clf.fit(X, y)
#     print(model.score(X, y))
#     # print(model, type(model))
#     feature_name = dataFrame.columns.values[:-1]
#     target_name = ['Yes', 'No']
#     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
#     tree.plot_tree(model, feature_names=feature_name,
#                class_names=target_name,
#                filled = True);
#     fig.savefig('imagename.png')


if __name__ == '__main__':
    main()





