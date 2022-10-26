import numpy as np
from sklearn.svm import SVC

PATH_X_TRAIN = "./data/X_train.csv"
PATH_Y_TRAIN = "./data/Y_train.csv"
PATH_X_TEST = "./data/X_test.csv"
PATH_Y_TEST = "./data/Y_test.csv"
IS_CONTAINHEAD = True
IS_NORMALIZE = False
NEED_RESHAPE = False
PENALTY = 0.001
THRESHOLD = 1e-19
STEP_SIZE = 0.0003
MAX_ITERATION = 6000


def change_dir():
    import os
    import sys
    os.chdir(sys.path[0])
    return sys.path[0]


def read_csv(path, option="x", need_reshape=NEED_RESHAPE):
    data = np.loadtxt(path, dtype=float, delimiter=',',
                      skiprows=int(IS_CONTAINHEAD))
    if option == "y":
        data[data == 0] = -1
        if need_reshape:
            data = np.reshape(data, (np.shape(data)[0], 1))
    return data


def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    for i in range(data.shape[0]):
        data[i, :] = (data[i, :] - mean) / std
    return data


def caculate_by_svm_sklearn(train_x, train_y, test_x, test_y, penalty=PENALTY, threshold=THRESHOLD, step_size=STEP_SIZE, max_iteration=MAX_ITERATION, is_normalize=IS_NORMALIZE):
    svm = SVC(kernel='linear')
    svm.fit(train_x, train_y)

    y_predict = svm.predict(test_x)
    correct_prediction = np.equal(y_predict, test_y)
    accuracy = np.mean(correct_prediction.astype(np.float))
    print("acc:", accuracy)

    return svm.intercept_[0], svm.coef_[0]


def main():
    data_x_train = read_csv(PATH_X_TRAIN, "x")
    data_y_train = read_csv(PATH_Y_TRAIN, "y")
    data_x_test = read_csv(PATH_X_TEST, "x")
    data_y_test = read_csv(PATH_Y_TEST, "y")
    print("read done!")
    print(caculate_by_svm_sklearn(data_x_train,
          data_y_train, data_x_test, data_y_test))


if __name__ == "__main__":
    change_dir()
    main()
