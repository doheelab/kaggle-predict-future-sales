# Lets first discuss what we are given and what we have to predict.
# About our dataset :
#
# We have in our training data :-
# 1. date - every date of items sold
# 2. date_block_num - this number given to every month
# 3. shop_id - unique number of every shop
# 4. item_id - unique number of every item
# 5. item_price - price of every item
# 6. item_cnt_day - number of items sold on a particular day
#
# We have in our testing data :-
# 1. ID - unique for every (shop_id,item_id) pair.
# 2. shop_id - unique number of every shop
# 3. item_id - unique number of every item
#
# Now what we have to predict ?
# we have to predict how many items of a type from each shop will be sold in a whole month.
# Our submission should have ID and item_cnt_month columns.

# 예측 하고자 하는 것?
# 2015년 10월에 각 상점들에서 각 상품 판매량을 예측하기

# What is our approach?
# our approach will be simple.
# Our features will be number of items sold in month from a shop excluding last month data because that will our labels,
# that we help our model learn to predict next sequence.
# And for testing will use number of items sold in month from a shop excluding first month like this dimension of our data remains same.
# Our model will predict the next sequence and that we will be our results.
# This is pretty simple approach but its good for start.
# Please try some different approaches also.
# And please let me know if I did something wrong. If you like it please vote it up.
#
# First of all as we know import required libraries

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.backend import dropout

sales_data = pd.read_csv("../input/sales_train.csv")
item_cat = pd.read_csv("../input/item_categories.csv")
items = pd.read_csv("../input/items.csv")
shops = pd.read_csv("../input/shops.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
test_data = pd.read_csv("../input/test.csv")


# Basic EDA
def basic_eda(df):
    print("----------TOP 5 RECORDS--------")
    print(df.head(5))
    print("----------INFO-----------------")
    print(df.info())
    print("----------Describe-------------")
    print(df.describe())
    print("----------Columns--------------")
    print(df.columns)
    print("----------Data Types-----------")
    print(df.dtypes)
    print("-------Missing Values----------")
    print(df.isnull().sum())
    print("-------NULL values-------------")
    print(df.isna().sum())
    print("-----Shape Of Data-------------")
    print(df.shape)


print("=============================Sales Data=============================")
basic_eda(sales_data)
print("=============================Test data=============================")
basic_eda(test_data)
print("=============================Item Categories=============================")
basic_eda(item_cat)
print("=============================Items=============================")
basic_eda(items)
print("=============================Shops=============================")
basic_eda(shops)
print("=============================Sample Submission=============================")
basic_eda(sample_submission)


# Preprocessing
def preprocess_data(sales_data):

    # we can see that 'date' column in sales_data is an object but if we want to manipulate
    # it or want to work on it someway then we have to convert it into datetime format
    sales_data["date"] = pd.to_datetime(sales_data["date"], format="%d.%m.%Y")

    # now we will create a pivot tabel by going so we get our data in desired form
    # we want get total count value of an item over the whole month for a shop
    # That why we made shop_id and item_id our indices and date_block_num our column
    # the value we want is item_cnt_day and used sum as aggregating function
    dataset = sales_data.pivot_table(
        index=["shop_id", "item_id"],  # 행 위치에 들어갈 열
        values=["item_cnt_day"],  # 데이터로 사용할 열
        columns=["date_block_num"],  # 열 위치에 들어갈 열
        fill_value=0,  # nan 값 채우기
        aggfunc="sum",  # 데이터 집계 함수
    )

    # lets reset our indices, so that data should be in way we can easily manipulate
    dataset.reset_index(inplace=True)
    dataset = pd.merge(test_data, dataset, on=["item_id", "shop_id"], how="left")

    # lets fill all NaN values with 0
    dataset.fillna(0, inplace=True)

    # we will drop shop_id and item_id because we do not need them
    # we are teaching our model how to generate the next sequence
    dataset.drop(["shop_id", "item_id", "ID"], inplace=True, axis=1)
    return dataset


dataset = preprocess_data(sales_data)


def train_test_split(dataset):

    # X we will keep all columns execpt the last one
    X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
    # the last column is our label
    y_train = dataset.values[:, -1:]

    # for test we keep all the columns execpt the first one
    X_test = np.expand_dims(dataset.values[:, 1:], axis=2)

    # lets have a look on the shape
    print(X_train.shape, y_train.shape, X_test.shape)
    return X_train, y_train, X_test


X_train, y_train, X_test = train_test_split(dataset)
dropout_rate = 0.4

# define model
def get_lstm_model():
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=64, input_shape=(33, 1)))
    lstm_model.add(Dropout(dropout_rate))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss="mse", optimizer="adam", metrics=["mean_squared_error"])
    lstm_model.summary()
    return lstm_model


# train model
def train_lstm_model(X_train, y_train):
    lstm_model = get_lstm_model()
    lstm_model.fit(X_train, y_train, batch_size=4096, epochs=10)
    return lstm_model


trained_lstm_model = train_lstm_model(X_train, y_train)

# submit results
submission_pfs = trained_lstm_model.predict(X_test)
# we will keep every value between 0 and 20
submission_pfs = submission_pfs.clip(0, 20)
# creating dataframe with required columns
submission = pd.DataFrame(
    {"ID": test_data["ID"], "item_cnt_month": submission_pfs.ravel()}
)
# creating csv file from dataframe
submission.to_csv(f"sub_pfs_{dropout_rate}.csv", index=False)

