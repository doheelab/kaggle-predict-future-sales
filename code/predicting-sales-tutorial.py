# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os

print(os.listdir("../input"))

train = pd.read_csv("../input/sales_train.csv")
items = pd.read_csv("../input/items.csv")
categories = pd.read_csv("../input/item_categories.csv")
shops = pd.read_csv("../input/shops.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")

# # EDA
# Let's look at the distribution of the training and test set to understand
# our dataset better and check for overlap, or lack thereof.

fig = plt.figure(figsize=(18, 9))
plt.subplots_adjust(hspace=0.5)

plt.subplot2grid((3, 3), (0, 0), colspan=3)
train["shop_id"].value_counts(normalize=True).plot(kind="bar", alpha=0.7)
plt.title("Shop ID Values in the Training Set (Normalized)")

plt.subplot2grid((3, 3), (1, 0))
train["item_id"].plot(kind="hist", alpha=0.7)
plt.title("Item ID Histogram")

plt.subplot2grid((3, 3), (1, 1))
train["item_price"].plot(kind="hist", alpha=0.7, color="orange")
plt.title("Item Price Histogram")

plt.subplot2grid((3, 3), (1, 2))
train["item_cnt_day"].plot(kind="hist", alpha=0.7, color="green")
plt.title("Item Count Day Histogram")

plt.subplot2grid((3, 3), (2, 0), colspan=3)
train["date_block_num"].value_counts(normalize=True).plot(kind="bar", alpha=0.7)
plt.title("Month (date_block_num) Values in the Training Set (Normalized)")

plt.show()

# The above graphs are a nice way to look at the raw distribtion of the test dataset. Here are some observations:
# 1. We have 60 'shop_id's but there is an uneven distribtution of these in the dataset. Four (<7%) of these shops make up ~25 percent of this dataset. These are shops (31, 25, 54, 28).
# 2. The Item IDs seem to have variations in frequency. We can't attribute a reason to this yet but we can inspect this further. Certain cateogeries are bound to sell better and maybe items under the same category are closer to each other as far as their ID distributions are concerned
# 3. From the vast empty spaces in the histograms of 'item_price' and 'item_cnt_day', we can infer that there are outliers in their distribution. **Let's write some simple code below to put a value to these outliers.**
# 4. Plotting the individual months from January 2013 to October 2015, it is interesting to see that the block month 12, corresponding to December 2013, had the highest number of sales. Month 23, which corresponds to December 2014, had the second highest number of sales. Shortly, we will use some better graphs to observe the monthly sale trends.

# ## Dealing with Outliers
train["item_id"].value_counts(ascending=False)[:5]

# Having a look at the item id 20949 that has been sold
# the most number of times, it is a plastic bag!
items.loc[items["item_id"] == 20949]
categories.loc[categories["item_category_id"] == 71]
test.loc[test["item_id"] == 20949].head(5)

# And just to satisfy my curiosity bowels, there is only one item under
# item category 71 and it is indeed present in the test set!
train["item_cnt_day"].sort_values(ascending=False)[:5]
train[train["item_cnt_day"] == 2169]

# So item 11373 was sold 2169 times at shop 12 on a single day in October.
# Let's inspect this outlier a bit more.
items[items["item_id"] == 11373]

# Using google translate, I understood that this item is related to point of delivery and the Russian shipment company 'Boxberry'. Let's look at some of the other daiy sales for this item
train[train["item_id"] == 11373].head(5)

# We can see that item 11373 usually has sells much less.
# I calculated the median of it's 'item_cnt_day' value to be 4. This confirms that the high value of 2169 is an anomaly and we should get rid of it.
train = train[train["item_cnt_day"] < 2000]
train["item_price"].sort_values(ascending=False)[:5]

# Another outlier! This time with 'item_price'. Let's inspect this a little further before we decide to remove it.
train[train["item_price"] == 307980]
items[items["item_id"] == 6066]

# Once again, using Google translate, I found that this is an antivirus sold to 522 people and the price is probabaly the cost of one installation times 522. Let's see if there are other transactions related to this software in our training set.
train[train["item_id"] == 6066]

# Nope! The outlier is the only transaction. It is justifiable to remove this observation from the training set.
train = train[train["item_price"] < 300000]

# Now, we will inspect the 'item_price' field for low priced outliers
train["item_price"].sort_values()[:5]

# And yes, there exists a value of -1. This is an error and we should inspect this observation further
train[train["item_price"] == -1]

# %% [markdown]
# Let's see if there are other observations for this item (2973) which can help us determine its price

# %%
train[train["item_id"] == 2973].head(5)

# %% [markdown]
# Indeed, this item seems to have been sold for quite a high price so its value of -1 should be changed. We should replace it with the median of its price, but this should be calculated for the shop (ID 12) for which this outlier exists. If there are other sales for this item in that shop for the same month (date_block_num 4), then we should calculate the median using that

# %%
price_correction = train[
    (train["shop_id"] == 32)
    & (train["item_id"] == 2973)
    & (train["date_block_num"] == 4)
    & (train["item_price"] > 0)
].item_price.median()
train.loc[train["item_price"] < 0, "item_price"] = price_correction

# %% [markdown]
# ## Test Set Distribution
# %% [markdown]
# Now, let's do the distribution analysis for the test set and see if we can spot any differences

# %%
fig = plt.figure(figsize=(18, 8))
plt.subplots_adjust(hspace=0.5)

plt.subplot2grid((3, 3), (0, 0), colspan=3)
test["shop_id"].value_counts(normalize=True).plot(kind="bar", alpha=0.7)
plt.title("Shop ID Values in the Test Set (Normalized)")

plt.subplot2grid((3, 3), (1, 0))
test["item_id"].plot(kind="hist", alpha=0.7)
plt.title("Item ID Histogram - Test Set")

plt.show()

# %% [markdown]
# 1. The Shop Id's are evenly spread out, unlike the training set. The font size of labels quickly tells me that there are certain Shop Id's missing in the test set as the bars in the training set 'shop_id' plot were more tightly packed.
# 2. While item id's in the histogram are binned, the spikes are less in the test set. The test set is much smaller in shape than the training set, and naturally, the frequency values are significantly lower. It is tough to be draw more insights from this histogram.
#
# It seems there might be some values of shop_id and item_id completely missing in the test set. Let's have a closer look and put some numbers or percentages to these missing values.
# %% [markdown]
# ## Shops Analysis

# %%
shops_train = train["shop_id"].nunique()
shops_test = test["shop_id"].nunique()
print("Shops in Training Set: ", shops_train)
print("Shops in Test Set: ", shops_test)

# %% [markdown]
# However, this doesn't mean that the training set contains all of the shops present in the test set. For that, we need to see if every element of the test set is present in the training set. Let's write some simple code to see if the test set list is a subset of the training set list.

# %%
shops_train_list = list(train["shop_id"].unique())
shops_test_list = list(test["shop_id"].unique())

flag = 0
if set(shops_test_list).issubset(set(shops_train_list)):
    flag = 1

if flag:
    print("Yes, list is subset of other.")
else:
    print("No, list is not subset of other.")

# %% [markdown]
# ### Duplicate Shops
# %% [markdown]
# Great, so all shops id's in the test set are also present in the training set. But there is an issue regarding duplicate shops that was talked about in the competition discussions. Let's have a look at the shops dataframe to identify this issue.

# %%
shops.T

# %% [markdown]
# Using [Google translate API](https://pypi.org/project/googletrans/) and measuring pair wise Levenstein distance with a python [library](https://github.com/luozhouyang/python-string-similarity), I generated a matrix plot to observe similarity with the shop names
#
# ![Matrix Plot](https://i.imgur.com/wIvpacw.png) ![Threshold](https://i.imgur.com/cH94yt2.png)
# %% [markdown]
# Upon examing the above matrixes and 'similar' shop names, one observes that shops start with their city name! This can be a useful feature. Also, 'shop_id's 0 and 1 are the same as 57 and 58 respectively. The only difference is the Russian word 'фран' attached to both shops 0 and 1. Shops 10 and 11 have the exact same name excpet the last character ('?' vs '2'). These duplicates should be removed!

# %%
train.loc[train["shop_id"] == 0, "shop_id"] = 57
test.loc[test["shop_id"] == 0, "shop_id"] = 57

train.loc[train["shop_id"] == 1, "shop_id"] = 58
test.loc[test["shop_id"] == 1, "shop_id"] = 58

train.loc[train["shop_id"] == 10, "shop_id"] = 11
test.loc[test["shop_id"] == 10, "shop_id"] = 11

# %% [markdown]
# ### Extracting City

# %%
cities = shops["shop_name"].str.split(" ").map(lambda row: row[0])
cities.unique()

# %% [markdown]
# Upon careful observation, the city Yakutsk is represented as *'!Якутск'* and *'Якутск'*. We should make sure they are the same so when we label encode these city names, they will fall under the same cateogory. For reference, the english tranlation of the cities is below.
#
# '!Yakutsk', 'Adygea', 'Balashikha', 'Volzhsky', 'Vologda', 'Voronezh', 'Outbound', 'Zhukovsky', 'Online', 'Kazan', 'Kaluga', 'Kolomna', 'Krasnoyarsk', 'Kursk', 'Moscow', 'Mytishchi', 'N.Novgorod', 'Novosibirsk', 'Omsk', 'RostovNaDonu', 'St.', 'Samara', 'Sergiev', 'Surgut', 'Tomsk', 'Tyumen', 'Ufa', 'Khimki', 'Digital', 'Chekhov', 'Yakutsk', 'Yaroslavl'
#
# You will notice we only extract the St. from St. Petersburg but that's okay as we will label encode these categorical variables.

# %%
shops["city"] = shops["shop_name"].str.split(" ").map(lambda row: row[0])
shops.loc[shops.city == "!Якутск", "city"] = "Якутск"


# %%
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit_transform(shops["city"])

# %% [markdown]
# Great, that worked! Let's just add these codes to the shops dataframe. We can get rid of *'shop_name'* and *'city'* while preserving the *'shop_id'* and *'city_label'*

# %%
shops["city_label"] = le.fit_transform(shops["city"])
shops.drop(["shop_name", "city"], axis=1, inplace=True)
shops.head()

# %% [markdown]
# ## Items Analysis
# %% [markdown]
# Let's perform a similar train-test analysis for item_ids

# %%
items_train = train["item_id"].nunique()
items_test = test["item_id"].nunique()
print("Items in Training Set: ", items_train)
print("Items in Test Set: ", items_test)

# %% [markdown]
# There are a lot many more items in the training set than there are in the test set. However, this doesn't mean that the training set contains all of the items in the test set. For that, we need to see if every element of the test set is present in the training set. Let's write some simple code to achieve this.

# %%
items_train_list = list(train["item_id"].unique())
items_test_list = list(test["item_id"].unique())

flag = 0
if set(items_test_list).issubset(set(items_train_list)):
    flag = 1

if flag:
    print("Yes, list is subset of other.")
else:
    print("No, list is not subset of other.")

# %% [markdown]
# Well then, this means there are certain items that are present in the test set but completely absent in the training set! Can we put a number to this to get an intuition?

# %%
len(set(items_test_list).difference(items_train_list))

# %% [markdown]
# There are 363 items that are present in the test set but completely absent in the training set. This doesn't mean that the sales prediction against those items must be zero as new items can be added to the market or we simply didn't possess the data for those items before. The fascinating questions pops though, how do you go about predicting them?
# %% [markdown]
# Before we do that, let's find out more about the 5100 items in the test set. What categories to they belong to? What categories do we not have to make predictions in the test set for

# %%
categories_in_test = items.loc[
    items["item_id"].isin(sorted(test["item_id"].unique()))
].item_category_id.unique()


# %%
# categories.loc[categories['item_category_id'].isin(categories_in_test)]
categories.loc[~categories["item_category_id"].isin(categories_in_test)].T

# %% [markdown]
# Using the same method, I generated a matrix plot to observe similarity with the shop names
#
# ![Category Matrix Plot](https://i.imgur.com/O5iTFj2.png) ![Category Matrix Threhsold](https://i.imgur.com/YZQrVRM.png)
# %% [markdown]
# ### Grouping Common Categories and Extracting Sub-Categories
# %% [markdown]
# Upon examing the above matrices, one can spot a trend of main and sub-categories. These are in the dataframe, before the hyphen. Let's extract them!

# %%
le = preprocessing.LabelEncoder()

main_categories = categories["item_category_name"].str.split("-")
categories["main_category_id"] = main_categories.map(lambda row: row[0].strip())
categories["main_category_id"] = le.fit_transform(categories["main_category_id"])

# Some items don't have sub-categories. For those, we will use the main category as a sub-category
categories["sub_category_id"] = main_categories.map(
    lambda row: row[1].strip() if len(row) > 1 else row[0].strip()
)
categories["sub_category_id"] = le.fit_transform(categories["sub_category_id"])


# %%
categories.head()

# %% [markdown]
# Now, let's generate sales data for each shop and item present in the training set. We should do this for each month, as the final prediction is for the monthly count of sales for a particular shop and item
# %% [markdown]
# ## Generating prodcuct of Shop-Item pairs for each month in the training data

# %%
train["date"] = pd.to_datetime(train["date"], format="%d.%m.%Y")


# %%
from itertools import product


# %%
# Testing generation of cartesian product for the month of January in 2013

shops_in_jan = train.loc[train["date_block_num"] == 0, "shop_id"].unique()
items_in_jan = train.loc[train["date_block_num"] == 0, "item_id"].unique()
jan = list(product(*[shops_in_jan, items_in_jan, [0]]))


# %%
print(len(jan))

# %% [markdown]
# As we can see, January 2013 contains 365,175 intersections of shops and items. Most of these will have no sales and we can verify this once we check our training set, which has been grouped by month, to see which 'items' x 'shops' combinations have a sale count associated with them.
#
# But before that, we need to generate this cartesian product for all 33 months in the training set. Before generating it for all months, I will generate it for February 2013, concatenate it with January 2013 and produce a dataframe.

# %%
# Testing generation of cartesian product for the month of February in 2013

shops_in_feb = train.loc[train["date_block_num"] == 1, "shop_id"].unique()
items_in_feb = train.loc[train["date_block_num"] == 1, "item_id"].unique()
feb = list(product(*[shops_in_feb, items_in_feb, [1]]))


# %%
print(len(feb))


# %%
cartesian_test = []
cartesian_test.append(np.array(jan))
cartesian_test.append(np.array(feb))


# %%
cartesian_test

# %% [markdown]
# Ran out of memory when trying to create a dataframe from january and february lists. The trick was to convert the lists to a numpy array. The handy numpy method 'vstack' will shape the cartesian_test object in the right manner, so we can convert it into a long form dataframe.

# %%
cartesian_test = np.vstack(cartesian_test)


# %%
cartesian_test_df = pd.DataFrame(
    cartesian_test, columns=["shop_id", "item_id", "date_block_num"]
)


# %%
cartesian_test_df.head()


# %%
cartesian_test_df.shape

# %% [markdown]
# Viola! That worked. Time to extend this to all months using some neat code

# %%
from tqdm import tqdm_notebook


def downcast_dtypes(df):
    """
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    """

    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    # Downcast
    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols] = df[int_cols].astype(np.int16)

    return df


# %%
months = train["date_block_num"].unique()


# %%
cartesian = []
for month in months:
    shops_in_month = train.loc[train["date_block_num"] == month, "shop_id"].unique()
    items_in_month = train.loc[train["date_block_num"] == month, "item_id"].unique()
    cartesian.append(
        np.array(
            list(product(*[shops_in_month, items_in_month, [month]])), dtype="int32"
        )
    )


# %%
cartesian_df = pd.DataFrame(
    np.vstack(cartesian),
    columns=["shop_id", "item_id", "date_block_num"],
    dtype=np.int32,
)


# %%
cartesian_df.shape

# %% [markdown]
# ## Aggregating sales to a monthly level and clipping target variable

# %%
x = (
    train.groupby(["shop_id", "item_id", "date_block_num"])["item_cnt_day"]
    .sum()
    .rename("item_cnt_month")
    .reset_index()
)
x.head()


# %%
x.shape

# %% [markdown]
# Now we need to merge our two dataframes. For the intersecting, we will simply put the values that exist in the dataframe x. For the remaining rows, we will sub in zero. Remember, the columns you want to merge on are the intersection of shop_id, item_id, and date_block_num

# %%
new_train = pd.merge(
    cartesian_df, x, on=["shop_id", "item_id", "date_block_num"], how="left"
).fillna(0)

# %% [markdown]
# By default, pandas fills the dataframes with NaN. That's why we use fillna to replace all NaN's with zero.

# %%
new_train["item_cnt_month"] = np.clip(new_train["item_cnt_month"], 0, 20)


# %%
del x
del cartesian_df
del cartesian
del cartesian_test
del cartesian_test_df
del feb
del jan
del items_test_list
del items_train_list
del train


# %%
new_train.sort_values(["date_block_num", "shop_id", "item_id"], inplace=True)
new_train.head()

# %% [markdown]
# ## Appending Test Set to Training Set
# %% [markdown]
# First, let's insert the date_block_num feature for the test set! Using insert method of pandas to place this new column at a specific index. This will allow us to concatenate the test set easily to the training set before we generate mean encodings and lag features

# %%
test.insert(loc=3, column="date_block_num", value=34)


# %%
test["item_cnt_month"] = 0


# %%
test.head()


# %%
new_train = new_train.append(test.drop("ID", axis=1))

# %% [markdown]
# ### Merging Shops, Items, Categories dataframes to add the city label, category_id, main category and sub-category feature

# %%
new_train = pd.merge(new_train, shops, on=["shop_id"], how="left")
new_train.head()


# %%
new_train = pd.merge(
    new_train, items.drop("item_name", axis=1), on=["item_id"], how="left"
)
new_train.head()


# %%
new_train = pd.merge(
    new_train,
    categories.drop("item_category_name", axis=1),
    on=["item_category_id"],
    how="left",
)
new_train.head()

# %% [markdown]
# ## Generating Lag Features and Mean-Encodings

# %%
def generate_lag(train, months, lag_column):
    for month in months:
        # Speed up by grabbing only the useful bits
        train_shift = train[["date_block_num", "shop_id", "item_id", lag_column]].copy()
        train_shift.columns = [
            "date_block_num",
            "shop_id",
            "item_id",
            lag_column + "_lag_" + str(month),
        ]
        train_shift["date_block_num"] += month
        train = pd.merge(
            train, train_shift, on=["date_block_num", "shop_id", "item_id"], how="left"
        )
    return train


# %%
del items
del categories
del shops
del test


# %%
new_train = downcast_dtypes(new_train)


# %%
import gc

gc.collect()

# %% [markdown]
# ### Lag for Target Variable

# %%
get_ipython().run_cell_magic(
    "time",
    "",
    "new_train = generate_lag(new_train, [1,2,3,4,5,6,12], 'item_cnt_month')",
)

# %% [markdown]
# ### Lag for monthly Item-Target mean

# %%
get_ipython().run_cell_magic(
    "time",
    "",
    "group = new_train.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()\nnew_train = pd.merge(new_train, group, on=['date_block_num', 'item_id'], how='left')\nnew_train = generate_lag(new_train, [1,2,3,6,12], 'item_month_mean')\nnew_train.drop(['item_month_mean'], axis=1, inplace=True)",
)

# %% [markdown]
# ### Lag for monthly Shop-Target mean

# %%
get_ipython().run_cell_magic(
    "time",
    "",
    "group = new_train.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()\nnew_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id'], how='left')\nnew_train = generate_lag(new_train, [1,2,3,6,12], 'shop_month_mean')\nnew_train.drop(['shop_month_mean'], axis=1, inplace=True)",
)

# %% [markdown]
# ### Lag for monthly Shop-Category mean

# %%
get_ipython().run_cell_magic(
    "time",
    "",
    "group = new_train.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('shop_category_month_mean').reset_index()\nnew_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')\nnew_train = generate_lag(new_train, [1, 2], 'shop_category_month_mean')\nnew_train.drop(['shop_category_month_mean'], axis=1, inplace=True)",
)

# %% [markdown]
# ### Lag for Monthly Main-Category mean

# %%
get_ipython().run_cell_magic(
    "time",
    "",
    "group = new_train.groupby(['date_block_num', 'main_category_id'])['item_cnt_month'].mean().rename('main_category_month_mean').reset_index()\nnew_train = pd.merge(new_train, group, on=['date_block_num', 'main_category_id'], how='left')\n\nnew_train = generate_lag(new_train, [1], 'main_category_month_mean')\nnew_train.drop(['main_category_month_mean'], axis=1, inplace=True)",
)

# %% [markdown]
# ### Lag for Monthly Sub-Category mean

# %%
get_ipython().run_cell_magic(
    "time",
    "",
    "group = new_train.groupby(['date_block_num', 'sub_category_id'])['item_cnt_month'].mean().rename('sub_category_month_mean').reset_index()\nnew_train = pd.merge(new_train, group, on=['date_block_num', 'sub_category_id'], how='left')\n\nnew_train = generate_lag(new_train, [1], 'sub_category_month_mean')\nnew_train.drop(['sub_category_month_mean'], axis=1, inplace=True)",
)


# %%
new_train.tail()

# %% [markdown]
# ### Add month feature!

# %%
new_train["month"] = new_train["date_block_num"] % 12

# %% [markdown]
# ### Add Holiday feature!

# %%
holiday_dict = {
    0: 6,
    1: 3,
    2: 2,
    3: 8,
    4: 3,
    5: 3,
    6: 2,
    7: 8,
    8: 4,
    9: 8,
    10: 5,
    11: 4,
}


# %%
new_train["holidays_in_month"] = new_train["month"].map(holiday_dict)

# %% [markdown]
# ### Russian Ruble Price per month!

# %%
# ruble_dollar = { 12: 33.675610, 13: 35.245171, 14: 36.195442, 15: 35.658811, 16: 34.918525, 17: 34.392044, 18: 34.684944, 19: 36.144526, 20: 37.951523, 21: 40.815324, 22: 46.257598, 23: 55.966912, 24: 63.676710, 25: 64.443511, 26: 60.261687, 27: 53.179035, 28: 50.682796, 29: 54.610770, 30: 57.155767, 31: 65.355082, 32: 66.950360, 33: 63.126499, 34: 65.083095, }


# %%
# new_train['ruble_value'] = new_train.date_block_num.map(ruble_dollar)

# %% [markdown]
# ### Russian Stock Exchange Trading Volume (in Trillions)

# %%
moex = {
    12: 659,
    13: 640,
    14: 1231,
    15: 881,
    16: 764,
    17: 663,
    18: 743,
    19: 627,
    20: 692,
    21: 736,
    22: 680,
    23: 1092,
    24: 657,
    25: 863,
    26: 720,
    27: 819,
    28: 574,
    29: 568,
    30: 633,
    31: 658,
    32: 611,
    33: 770,
    34: 723,
}


# %%
new_train["moex_value"] = new_train.date_block_num.map(moex)


# %%
new_train = downcast_dtypes(new_train)

# %% [markdown]
# # Training

# %%
import xgboost as xgb


# %%
new_train = new_train[new_train.date_block_num > 11]

# x_train = new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1)
# y_train = new_train[new_train.date_block_num < 33]['item_cnt_month']

# x_valid = new_train[new_train.date_block_num == 33].drop(['item_cnt_month'], axis=1)
# y_valid = new_train[new_train.date_block_num == 33]['item_cnt_month']

# x_test = new_train[new_train.date_block_num == 34].drop(['item_cnt_month'], axis=1)


# %%
import gc

gc.collect()


# %%
def fill_na(df):
    for col in df.columns:
        if ("_lag_" in col) & (df[col].isnull().any()):
            df[col].fillna(0, inplace=True)
    return df


new_train = fill_na(new_train)

new_train = new_train.reset_index(drop=True)
new_train.to_feather("new_train_baller")
# %%
def xgtrain():
    regressor = xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.5,
        colsample_bytree=0.5,
    )

    regressor_ = regressor.fit(
        new_train[new_train.date_block_num < 33]
        .drop(["item_cnt_month"], axis=1)
        .values,
        new_train[new_train.date_block_num < 33]["item_cnt_month"].values,
        eval_metric="rmse",
        eval_set=[
            (
                new_train[new_train.date_block_num < 33]
                .drop(["item_cnt_month"], axis=1)
                .values,
                new_train[new_train.date_block_num < 33]["item_cnt_month"].values,
            ),
            (
                new_train[new_train.date_block_num == 33]
                .drop(["item_cnt_month"], axis=1)
                .values,
                new_train[new_train.date_block_num == 33]["item_cnt_month"].values,
            ),
        ],
        verbose=True,
        early_stopping_rounds=50,
    )
    return regressor_


# %%
get_ipython().run_cell_magic("time", "", "regressor_ = xgtrain()")


# %%
predictions = regressor_.predict(
    new_train[new_train.date_block_num == 34].drop(["item_cnt_month"], axis=1).values
)


# %%
from matplotlib import rcParams

rcParams["figure.figsize"] = 11.7, 8.27

cols = new_train.drop("item_cnt_month", axis=1).columns
plt.barh(cols, regressor_.feature_importances_)
plt.show()


# %%
submission["item_cnt_month"] = predictions


# %%
submission.to_csv("sales_faster_learn.csv", index=False)


# %%
from IPython.display import FileLinks

FileLinks(".")

# %% [markdown]
# Code to render Altair Charts

# %%
import json  # need it for json.dumps
from IPython.display import HTML

# Create the correct URLs for require.js to find the Javascript libraries
vega_url = "https://cdn.jsdelivr.net/npm/vega@" + alt.SCHEMA_VERSION
vega_lib_url = "https://cdn.jsdelivr.net/npm/vega-lib"
vega_lite_url = "https://cdn.jsdelivr.net/npm/vega-lite@" + alt.SCHEMA_VERSION
vega_embed_url = "https://cdn.jsdelivr.net/npm/vega-embed@3"
noext = "?noext"

altair_paths = {
    "vega": vega_url + noext,
    "vega-lib": vega_lib_url + noext,
    "vega-lite": vega_lite_url + noext,
    "vega-embed": vega_embed_url + noext,
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {paths}
}});
"""

# Define the function for rendering
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):
        """Render an altair chart directly via javascript.
        
        This is a workaround for functioning export to HTML.
        (It probably messes up other ways to export.) It will
        cache and autoincrement the ID suffixed with a
        number (e.g. vega-chart-1) so you don't have to deal
        with that.
        """
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + "-" + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)

    # Cache will stay defined and keep track of the unique div Ids
    return wrapped


@add_autoincrement
def render_alt(chart, id="vega-chart"):
    # This below is the javascript to make the chart directly using vegaEmbed
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vegaEmbed) {{
        const spec = {chart};     
        vegaEmbed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
    }});
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart)
            if isinstance(chart, dict)
            else chart.to_json(indent=None),
        )
    )


HTML(
    "".join(
        ("<script>", workaround.format(paths=json.dumps(altair_paths)), "</script>")
    )
)


# %%
alt.data_transformers.enable("default", max_rows=None)

# %% [markdown]
# 1. Plotting Ideas: Number of unique items sold each month
# 2. Sales graph, Revenue graph

# %%
train = pd.read_csv("../input/sales_train.csv")


# %%
alt_df = (
    train.groupby("date_block_num")["item_cnt_day"].sum().rename("sum").reset_index()
)
alt_df.head(2)


# %%
# Generate a mapping for date_block_num to month and year.
dict_date_block_num = {
    0: "January 2013",
    1: "February 2013",
    2: "March 2013",
    3: "April 2013",
    4: "May 2013",
    5: "June 2013",
    6: "July 2013",
    7: "August 2013",
    8: "September 2013",
    9: "October 2013",
    10: "November 2013",
    11: "December 2013",
    12: "January 2014",
    13: "February 2014",
    14: "March 2014",
    15: "April 2014",
    16: "May 2014",
    17: "June 2014",
    18: "July 2014",
    19: "August 2014",
    20: "September 2014",
    21: "October 2014",
    22: "November 2014",
    23: "December 2014",
    24: "January 2015",
    25: "February 2015",
    26: "March 2015",
    27: "April 2015",
    28: "May 2015",
    29: "June 2015",
    30: "July 2015",
    31: "August 2015",
    32: "September 2015",
    33: "October 2015",
}


# %%
alt_df["date_block_num"].replace(dict_date_block_num, inplace=True)
alt_df.head(2)


# %%
split = alt_df["date_block_num"].str.split(" ", n=1, expand=True)


# %%
alt_df_two = alt_df.copy()
alt_df_two["month"] = split[0]
alt_df_two["year"] = split[1]
alt_df_two.head()


# %%
chart1 = (
    alt.Chart(alt_df_two)
    .mark_area()
    .encode(
        alt.X("date_block_num:T", axis=alt.Axis(labelAngle=-45), title="Time"),
        alt.Y("sum:Q", title="Items Sold"),
        alt.Color(
            "month:N",
            scale=alt.Scale(scheme="category20"),
            sort=["Januray"],
            title="Month",
        ),
    )
    .properties(
        width=1000,
        title="Kaggle Data Science Competition: Sales Prediction in Russian Stores",
    )
)


# %%
alt.themes.enable("opaque")


# %%
render_alt(chart1)


# %%
chart2 = (
    alt.Chart(alt_df_two)
    .mark_bar()
    .encode(alt.X("year:N"), alt.Y("sum(sum):Q"), alt.Color("year:N"))
    .properties(width=1000)
)


# %%
render_alt(chart2)


# %%
alt_df_three = (
    train.groupby(["date_block_num", "shop_id"])["item_cnt_day"]
    .sum()
    .rename("sum")
    .reset_index()
)
alt_df_three["date_block_num"].replace(dict_date_block_num, inplace=True)

split = alt_df_three["date_block_num"].str.split(" ", n=1, expand=True)
alt_df_three["month"] = split[0]
alt_df_three["year"] = split[1]

alt_df_three.head(2)


# %%
chart3 = (
    alt.Chart(alt_df_three)
    .mark_rect()
    .encode(alt.Y("month"), alt.X("shop_id:N"), alt.Color("mean(sum):Q"))
)

chart4 = (
    alt.Chart(alt_df_three)
    .mark_rect()
    .encode(alt.Y("year"), alt.X("shop_id:N"), alt.Color("mean(sum):Q"))
)

chart = chart3 & chart4


# %%
render_alt(chart)

