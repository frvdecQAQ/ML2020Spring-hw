# -*- coding:utf-8 -*- 
import sys
import numpy as np
import pandas as pd

month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def next_date(date):
    date_num = [int(i) for i in date.split('/')]
    #print(date_num)
    date_num[2] += 1
    if date_num[2] > month[date_num[1]-1]:
        date_num[2] = 0
        date_num[1] += 1
    return '/'.join([str(i) for i in date_num])

def pre_process(df):
    X = np.ones(shape=(5652, 162), dtype='float64')
    Y = np.zeros(shape=(5652, 1), dtype='float64')

    date_list = np.unique(np.array(df['日期']))
    hour_list = [[str(j) for j in range(i, i+9)] for i in range(0, 15)]
    cnt = 0

    for date in date_list:
        tmp_df = df[df['日期'] == date]
        for hour in range(0, 15):
            tmp = np.array(tmp_df[hour_list[hour]])
            tmp[tmp == 'NR'] = '0'
            tmp = tmp.astype('float64')
            X[cnt] = tmp.reshape(162)

            tmp = np.array(tmp_df[tmp_df['測項'] == 'PM2.5'][str(hour+9)])
            tmp[tmp == 'NR'] = '0'
            tmp = tmp.astype('float64')
            Y[cnt] = tmp
            cnt += 1
        
        nxt_df = df[df['日期'] == next_date(date)]
        if not nxt_df.empty:
            for hour in range(15, 24):
                t1 = np.array(tmp_df[[str(i) for i in range(hour, 24)]])
                t2 = np.array(nxt_df[[str(i) for i in range(0, hour-15)]])
                tmp = np.hstack((t1, t2))
                tmp[tmp == 'NR'] = '0'
                tmp = tmp.astype('float64')
                X[cnt] = tmp.reshape(162)

                tmp = np.array(nxt_df[nxt_df['測項'] == 'PM2.5'][str(hour-15)])
                tmp[tmp == 'NR'] = '0'
                tmp = tmp.astype('float64')
                Y[cnt] = tmp
                cnt += 1
    print(cnt)
    return X, Y

def normalization(x, x_mean, x_std):
    x = (x-x_mean)/x_std
    x = np.hstack((x, np.ones(shape = (x.shape[0], 1))))
    return x

def split(X, Y, test_alpha):
    shuffled_indices = np.random.permutation(X.shape[0])
    test_set_size = int(X.shape[0]*test_alpha)
    return X[shuffled_indices[:test_set_size]], X[shuffled_indices[test_set_size:]],\
           Y[shuffled_indices[:test_set_size]], Y[shuffled_indices[test_set_size:]]

def simple_train(X_train, Y_train, X_test, Y_test):
    learning_rate = 0.01
    theta = np.random.zeros((163, 1))
    step = 0
    while step < 1000:
        step += 1
        Y_hat = np.dot(X_train, theta)
        rmse_train = np.sqrt(np.mean(np.square(Y_hat-Y_train)))
        delta = np.dot(X_train.T, Y_hat-Y_train)/Y.size
        theta = theta-learning_rate*delta
        delta_mean = np.mean(np.square(delta))

        Y_hat = np.dot(X_test, theta)
        rmse_test = np.sqrt(np.mean(np.square(Y_hat-Y_test)))

        if step%100 == 0:
            print(f'step = {step}, mse_train = {rmse_train}, mse_test = {rmse_test}, delta_mean = {delta_mean}')
    
    return theta

def adagrad_train(X_train, Y_train, X_test, Y_test):
    learning_rate = 100
    eps = 1e-12
    theta = np.zeros((163, 1))
    #batch = 256
    step = 0
    delta_sum = np.zeros((163, 1))
    while step < 10000:
        step += 1
        #shuffled_indices = np.random.permutation(X_train.shape[0])
        #now_x = X_train[shuffled_indices[:batch]]
        #now_y = Y_train[shuffled_indices[:batch]]

        y_hat = np.dot(X_train, theta)
        rmse_train = np.sqrt(np.mean(np.square(y_hat-Y_train)))
        delta = np.dot(X_train.T, y_hat-Y_train)/Y_train.size
        delta_sum = delta_sum+np.square(delta)
        theta = theta-learning_rate*delta/(np.sqrt(delta_sum+eps))
        delta_mean = np.mean(np.square(delta))

        Y_hat = np.dot(X_test, theta)
        rmse_test = np.sqrt(np.mean(np.square(Y_hat-Y_test)))

        if step%100 == 0:
            print(f'step = {step}, mse_train = {rmse_train}, mse_test = {rmse_test}, delta_mean = {delta_mean}')

    return theta

def ans_feature(df):
    X = np.zeros((240, 162))
    column_select = range(2, 11)

    for i in range(0, 240):
        tmp = np.array(df[df[0] == 'id_'+str(i)][column_select])
        tmp[tmp == 'NR'] = '0'
        tmp = tmp.astype('float64').reshape(162)
        X[i] = tmp
    return X

if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv', encoding='big5')
    X, Y = pre_process(df)
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    X = normalization(X, x_mean, x_std)
    X_test, X_train, Y_test, Y_train = split(X, Y, 0.2)

    #simple_train(X_train, Y_train, X_test, Y_test)
    theta = adagrad_train(X_train, Y_train, X_test, Y_test)
    np.save('theta', theta)

    df = pd.read_csv('../data/test.csv', header=None)
    X_ans = ans_feature(df)
    X_ans = normalization(X_ans, x_mean, x_std)
    predict = np.dot(X_ans, theta).reshape(240)
    ans_df = pd.DataFrame({'id':['id_'+str(i) for i in range(0, 240)], 'value': predict})
    ans_df.to_csv('submission.csv', index=False)

