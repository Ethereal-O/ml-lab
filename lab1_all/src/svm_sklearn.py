import numpy as np

PATH_X_TRAIN = "./data/X_train.csv"
PATH_Y_TRAIN = "./data/Y_train.csv"
ISCONTAINHEAD = True
PENALTY = 0.01
THRESHOLD = 1e-19
STEP_SIZE = 0.01
# STEP_SIZE = 1e-4
MAX_ITERATION = 6000


def change_dir():
    import os
    import sys
    os.chdir(sys.path[0])
    return sys.path[0]


def read_csv(path):
    return np.loadtxt(path, dtype=float, delimiter=',', skiprows=int(ISCONTAINHEAD))


def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    for i in range(data.shape[0]):
        data[i, :] = (data[i, :] - mean) / std
    return data


def caculate_by_gradient_decent(data_x, data_y, penalty=PENALTY, threshold=THRESHOLD, step_size=STEP_SIZE, max_iteration=MAX_ITERATION):
    data_x = normalize_data(data_x)
    data_num = np.shape(data_x)[0]
    feature_dim = np.shape(data_x)[1]
    w = np.random.rand(feature_dim, 1)*0.01
    w_0 = np.random.rand(1)*0.01

    it = 1
    th = 0.1
    while it < max_iteration and th > threshold:
        a = np.tile(w_0, [data_num, 1])+data_x@w
        ksi = a*data_y
        index = (ksi < 1)[:, 0]

        dw = np.zeros([feature_dim, 1])
        dw_0 = 0
        # for w_other
        dw = -np.transpose(data_x[index])@data_y[index]
        dw = dw/data_num+penalty*w
        # for w_0
        dw_0 = -sum(data_y[index])/data_num

        w_0_ = w_0-step_size*dw_0
        w_ = w-step_size*dw

        th = np.sum(np.square(w_ - w)) + np.square(w_0_ - w_0)
        it = it + 1

        w = w_
        w_0 = w_0_

        if it % 200 == 0:
            y_predict = np.tile(w_0, [data_num, 1])+data_x@w
            y_predict = [1 if i > 0.5 else 0 for i in y_predict ]
            correct_prediction = abs(y_predict-data_y)
            accuracy = np.mean(correct_prediction.astype(np.float))
            print("epoch:", it, "acc:", accuracy, "th:", th)
            # print(y_predict)
    return w_0, w


def main():
    data_x_train = read_csv(PATH_X_TRAIN)
    data_y_train = read_csv(PATH_Y_TRAIN)
    print("read done!")
    data_y_train = np.reshape(data_y_train, (np.shape(data_y_train)[0], 1))
    print(caculate_by_gradient_decent(data_x_train, data_y_train))


if __name__ == "__main__":
    change_dir()
    main()
