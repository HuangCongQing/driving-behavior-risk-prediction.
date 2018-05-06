# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import datetime
import warnings

warnings.filterwarnings('ignore')


import lightgbm as lgb
import time

import numpy as np


path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def timestamp_datetime(value):
    fmt = '%Y-%m-%d %H:%M:%S'

    value = time.localtime(value)
    dt = time.strftime(fmt, value)
    return dt


def get_feat(train):
    train['TIME'] = train['TIME'].apply(lambda x: timestamp_datetime(x), 1)
    train['TIME'] = train['TIME'].apply(lambda x: str(x)[:13], 1)
    train = train.sort_values(by=["TERMINALNO", 'TIME'])
    train.index = range(len(train))
    train['hour'] = train.TIME.apply(lambda x: str(x)[11:13], 1)
    train['hour'] = train['hour'].astype(int)

    train['is_hour_0'] = train.hour.apply(lambda x: 1 if x == 0 else 0, 1)
    train['is_hour_1'] = train.hour.apply(lambda x: 1 if x == 1 else 0, 1)
    train['is_hour_2'] = train.hour.apply(lambda x: 1 if x == 2 else 0, 1)
    train['is_hour_3'] = train.hour.apply(lambda x: 1 if x == 3 else 0, 1)
    train['is_hour_4'] = train.hour.apply(lambda x: 1 if x == 4 else 0, 1)
    train['is_hour_21'] = train.hour.apply(lambda x: 1 if x == 21 else 0, 1)
    train['is_hour_22'] = train.hour.apply(lambda x: 1 if x == 22 else 0, 1)
    train['is_hour_23'] = train.hour.apply(lambda x: 1 if x == 23 else 0, 1)

    train_hour = train.groupby(['TERMINALNO', 'TIME'], as_index=False).count()
    train_hour.TIME = train_hour.TIME.apply(lambda x: str(x)[:10], 1)
    train_day = train_hour.groupby(
        ['TERMINALNO', 'TIME'], as_index=False).count()

    train_hour_count = train_day.groupby('TERMINALNO')['LONGITUDE'].agg(
        {"hour_count_max": "max", "hour_count_min": "min", "hour_count_mean": "mean", "hour_count_std": "std", "hour_count_skew": "skew"}).reset_index()

    train_hour_first = train.groupby(
        ['TERMINALNO', 'TIME'], as_index=False).first()
    train_hour_first.TIME = train_hour_first.TIME.apply(
        lambda x: str(x)[:10], 1)
    train_day_sum = train_hour_first.groupby(
        ['TERMINALNO', 'TIME'], as_index=False).sum()
    train_day_sum['hour_count'] = train_day['LONGITUDE']

    train_day_sum['night_drive_count'] = train_day_sum.apply(lambda x: x['is_hour_0'] + x['is_hour_1'] +
                                                             x['is_hour_2']+x['is_hour_3']+x['is_hour_4'] +
                                                             x['is_hour_21']+x['is_hour_22']+x['is_hour_23'], 1)

    train_day_sum['night_delta'] = train_day_sum['night_drive_count'] / \
        train_day_sum['hour_count']

    train_day_sum['is_night'] = train_day_sum['night_drive_count'].apply(
        lambda x: 1 if x != 0 else 0, 1)
    train_hour_count['night__day_delta'] = train_day_sum.groupby(['TERMINALNO'], as_index=False).sum(
    )['is_night']/(train_day_sum.groupby(['TERMINALNO'], as_index=False).count()['HEIGHT'])

    train_night_count = train_day_sum.groupby('TERMINALNO')['night_delta'].agg(
        {"night_count_max": "max", "night_count_min": "min", "night_count_mean": "mean", "night_count_std": "std", "night_count_skew": "skew"}).reset_index()

    train_data = pd.merge(
        train_hour_count, train_night_count, on="TERMINALNO", how="left")

    return train_data


def get_label(dataset, train):
    dataset['label'] = train.groupby(
        'TERMINALNO')['Y'].last().reset_index()['Y']
    return dataset


def save(test, pred, name):

    dt = datetime.datetime.now().strftime("%Y%m%d")
    test['Id'] = test['TERMINALNO']
    test['Pred'] = pred
    test[['Id', 'Pred']].to_csv(
        path_test_out+"%s_%s.csv" % (dt, name), index=False)


def fit_model(train, test, params, num_round, early_stopping_rounds):
    features = [x for x in train.columns if x not in ["TERMINALNO", 'label']]
    label = 'label'

    dtrain = lgb.Dataset(train[features], label=train[label])
    model = lgb.train(params, dtrain, num_boost_round=1500,

                      )
    t_pred = model.predict(test[features])
    return t_pred


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    train_data = pd.read_csv(path_train)
    train_data = train_data.ix[:15000000, :]
    test_data = pd.read_csv(path_test)

    # print("========building the dataset========")
    train_X = get_feat(train_data)
    train = get_label(train_X, train_data)
    print(train.shape)
    test = get_feat(test_data)
    print(test.shape)

    params = {
        'boosting_type': 'gbdt',
        # 'metric': 'auc',
        # 'is_unbalance': 'True',
        'learning_rate': 0.01,
        'verbose': 0,
        'num_leaves': 32,
        # 'max_depth': 5,
        # "max_bin": 10,
        #           "reg_lambda":11,
        #          "reg_alpha":10,
        'objective': 'regression',
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,  # 0.9是目前最优的
        'bagging_freq': 1,  # 3是目前最优的
        #             'min_data': 500,
        'seed': 1024,
        'nthread': 12,
        # 'silent': True,
    }

    num_round = 1500
    early_stopping_rounds = 100
    # print("============training model===========")

    rlt_pred = fit_model(train, test, params, num_round, early_stopping_rounds)

    save(test, rlt_pred, "lgb")
    endtime = datetime.datetime.now()
    print("use time: ", (endtime - starttime).seconds, " s")
    # print("===========done============")
