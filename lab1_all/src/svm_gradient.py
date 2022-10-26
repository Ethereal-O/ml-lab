import numpy as np

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


def caculate_by_gradient_decent(train_x, train_y, test_x, test_y, penalty=PENALTY, threshold=THRESHOLD, step_size=STEP_SIZE, max_iteration=MAX_ITERATION, is_normalize=IS_NORMALIZE):
    if (is_normalize):
        train_x = normalize_data(train_x)
        test_x = normalize_data(test_x)
    data_num = np.shape(train_x)[0]
    feature_dim = np.shape(train_x)[1]
    w = np.random.rand(feature_dim, 1)*0.01
    w_0 = np.random.rand(1)*0.01

    it = 1
    th = 0.1
    while it < max_iteration and th > threshold:
        a = np.tile(w_0, [data_num, 1])+train_x@w
        ksi = a*train_y
        index = (ksi < 1)[:, 0]

        dw = np.zeros([feature_dim, 1])
        dw_0 = 0
        # for w_other
        dw = -np.transpose(train_x[index])@train_y[index]
        dw = dw/data_num+penalty*w
        # for w_0
        dw_0 = -sum(train_y[index])/data_num

        w_0_ = w_0-step_size*dw_0
        w_ = w-step_size*dw

        th = np.sum(np.square(w_ - w)) + np.square(w_0_ - w_0)
        it = it + 1

        w = w_
        w_0 = w_0_

        if it % 200 == 0:
            y_predict = np.tile(w_0, [np.shape(test_x)[0], 1])+test_x@w
            y_predict = [[1] if i > 0 else [-1] for i in y_predict]
            correct_prediction = np.equal(y_predict, test_y)
            accuracy = np.mean(correct_prediction.astype(np.float))
            print("epoch:", it, "acc:", accuracy, "th:", th)
    return w_0, w


def main():
    data_x_train = read_csv(PATH_X_TRAIN, "x")
    data_y_train = read_csv(PATH_Y_TRAIN, "y")
    data_x_test = read_csv(PATH_X_TEST, "x")
    data_y_test = read_csv(PATH_Y_TEST, "y")
    print("read done!")
    print(caculate_by_gradient_decent(data_x_train,
          data_y_train, data_x_test, data_y_test))


if __name__ == "__main__":
    change_dir()
    main()
