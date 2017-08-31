# =========================== Setup tools =====================================
import numpy as np
import pandas as pd
import csv
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sklearn.metrics as metrics
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn import metrics
from sklearn import datasets
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
from IPython.core.display import HTML

# set the max rows into 15 for display
pd.set_option('display.height', 15)
pd.set_option('display.max_Rows', 15)

listings_file = 'listings.csv'

# set desired columns to extract
columns = ['price',
           'summary',
           'neighbourhood_cleansed',
           'property_type',
           'room_type',
           'price',
           'number_of_reviews',
           'instant_bookable',
           'review_scores_rating',
           'beds',
           'bedrooms',
           'bathrooms',
           'accommodates',
           'amenities',
           'cancellation_policy',
           'reviews_per_month',
           'latitude',
           'longitude']

# ========================== Prepare data =====================================

# read the listing file downloaded from http://insideairbnb.com/
listing = pd.read_csv(listings_file, usecols=columns)

# replace NaN values with 0
listing.fillna(0, inplace=True)

# extract the price from the table
price = listing['price']
prices = []

# then change the price from strings to float numbers
for element in price:
    element = float(element[1:].replace(',',''))
    prices.append(element)

# replace the current column into a new one for future use
listing['price'] = prices

# filter out listings with invalid numbers
listing = listing[listing.price > 0]
listing = listing[listing.bedrooms > 0]
listing = listing[listing.beds > 0]
listing = listing[listing.review_scores_rating > 0]
listing = listing[listing.reviews_per_month > 0]
listing = listing[listing.accommodates > 0]
listing = listing[listing.bathrooms > 0]

# first 5 of the listing file
listing[0:5]

# get location for future use
location = listing[['latitude', 'longitude']]

# =================== Generate / Visualize the results ========================

plt.figure(figsize=(12,12))

# the longitude and latitude limits of Seattle
map_extent = [-122.46, 47.48, -122.22, 47.73]
themap = Basemap(llcrnrlon=map_extent[0], 
                 llcrnrlat=map_extent[1],
                 urcrnrlon=map_extent[2], 
                 urcrnrlat=map_extent[3],
                 projection='gall',
                 resolution='f', epsg=4326)

# create outlines for the basemap
themap.drawcoastlines()
themap.drawcountries()
themap.fillcontinents(color = 'gainsboro')
themap.drawmapboundary(fill_color='steelblue')

# first split each type of property
home = listing[(listing.room_type == 'Entire home/apt')]
private = listing[(listing.room_type == 'Private room')]
shared = listing[(listing.room_type == 'Shared room')]

# then split them based on longitude and latitude
a, b = themap(home['longitude'], home['latitude'])
c, d = themap(private['longitude'], private['latitude'])
e, f = themap(shared['longitude'], shared['latitude'])

# plot using different colors
themap.plot(a, b, 'o',                   
            color='Red',      
            markersize=4)
themap.plot(c, d, 'o',                   
            color='Green',       
            markersize=4)
themap.plot(e, f, 'o',                    
            color='Blue',         
            markersize=4)

# use counter to show how many listings in each neighborhood
nh = Counter(listing['neighbourhood_cleansed'])
print nh

nh_df = pd.DataFrame.from_dict(nh, orient='index').sort_values(by=0)
nh_df.plot(kind='bar', 
           color = 'LightBlue', 
           figsize =(15,8), 
           title = 'Seattle Neighborhood Frequency', 
           legend = False)

# calculate average price of all listings in Seattle
average_price = sum(listing.price) / float(len(listing.price))
print 'Average price of all listsing in Seattle is ' + str(average_price)

# extract the names
neighborhood_names = list(nh.keys())

# 2 column table of neighborhood names and prices
nh_prices = listing[['neighbourhood_cleansed', 'price']]
nh_prices.columns = ['neighbourhood', 'price']

# pick out the rows which have neighborhood names with 400+ listings.
nh_prices = nh_prices[nh_prices['neighbourhood'].isin(neighborhood_names)]

# group by neighbourhood and then aggreate the prices based on mean
nh_prices_group = nh_prices.groupby('neighbourhood')
nh_prices = nh_prices_group['price'].agg(np.mean)

# turn dictionary's keys and values into a table for easy read
nh_prices = nh_prices.reset_index()
nh_prices['number of listings'] = nh.values()

# show average price for each neighborhood
print 'Average price for each neighborhood is '
print str(nh_prices)

# graph each neighborhood based on their prices
p = nh_prices.sort_values(by = 'price')
p.plot(x = "neighbourhood",
       y = "price",
       kind='bar', 
       color = 'LightBlue', 
       figsize =(15,8), 
       title = 'Price per Neighorhood', 
       legend = False)
# calculate and plot the percentage of room types
room = listing.room_type
r = Counter(room)

room_df = pd.DataFrame.from_dict(r, orient='index').sort_values(by=0)
room_df.columns = ['room_type']
room_df.plot.pie(y = 'room_type', 
                 colormap = 'Blues_r', 
                 figsize=(8,8), 
                 fontsize = 20, autopct = '%.2f',
                 legend = False,
                 title = 'Room Type Distribution')

# calculate and plot the percentage the property types
property = listing.property_type
p = Counter(property)

property_df = pd.DataFrame.from_dict(p, orient='index').sort_values(by=0)
property_df.columns = ['property_type']
property_df.plot.bar(y= 'property_type', 
                     color = 'LightBlue',
                     fontsize = 20,
                     legend = False,
                     figsize= (13, 7),
                     title = "Property type distribution")

# break down property and room types by price
prop_room = listing[['property_type', 'room_type', 'price']]

# first ten of the table
prop_room[0:10]

# Group by porperty and room type
prop_room_group = prop_room.groupby(['property_type', 'room_type']).mean()

# reset the index in order to turn the lists into a readable table
p = prop_room_group.reset_index()

# pivot the table based on the 3 factors
p = p.pivot('property_type', 'room_type', 'price')

# replace the NaN values with 0
p.fillna(0.00, inplace=True)

p

# analyze the effect of summary 
summary = listing[['summary']]

# gets rid of NaN values
summary = summary[pd.notnull(summary['summary'])]
summary

# use counter to find the most used words when people describe their apartments
words = []

for detail in summary['summary']:
    if detail != 0:
        for word in detail.split():
            words.append(word)

# turning the list into a counter
words = Counter(words)
word_count = pd.DataFrame.from_dict(words, orient='index').sort_values(by=0)

# rename the column
word_count.columns = ['summary']

# sort from highest to lowest
word_count = word_count.sort_values(by=['summary'], ascending=False)
word_count

# analyze the price and number of reviews
price_review = listing[['number_of_reviews', 'price']].sort_values(by = 'price')

price_review.plot(x = 'price', 
                  y = 'number_of_reviews', 
                  style = 'o',
                  figsize =(12,8),
                  legend = False,
                  title = 'Reviews based on Price')

# total number of reviews 
sum(listing.number_of_reviews)

# average number of reviews per listing
np.mean(listing.reviews_per_month)

# find cancellation policy distribution
cancel = listing.cancellation_policy
c = Counter(cancel)

# clean up small values
c.pop("super_strict_30", None)
c.pop("super_strict_60", None)
cancel_df = pd.DataFrame.from_dict(c, orient='index').sort_values(by=0)
cancel_df.columns = ['Cancellation Policy']
cancel_df.plot.pie(y = 'Cancellation Policy',
                   colormap = 'Blues_r',
                   figsize=(8,8), 
                   fontsize = 20, 
                   autopct = '%.2f',
                   legend = False,
                   title = "Cancellation Policy Distribution")

# ======================== Predict future trends ==============================
# combine analyzed results and predict future price
data = listing[['price',
           'room_type',
           'accommodates',
           'bathrooms',
           'bedrooms',
           'beds',
           'review_scores_rating',
           'instant_bookable',
           'cancellation_policy',
           'amenities']]

cancel_policy = pd.get_dummies(data.cancellation_policy).astype(int)
instant_booking = pd.get_dummies(data.instant_bookable, prefix = 'instant_booking').astype(int)
room_type = pd.get_dummies(data.room_type).astype(int)

# ib has 2 columns, just drop one of them.
instant_booking = instant_booking.drop('instant_booking_f', axis = 1)

# drop the original columns and replace them with indicator columns
data = data.drop(['cancellation_policy', 'instant_bookable', 'room_type'], axis = 1)
data = pd.concat((data, cancel_policy, instant_booking, room_type), axis = 1)

# split the amenities list to draw out how many amenities each listing has
amenities_list = []

for element in data.amenities:
    element = element[1:]
    element = element[:-1]
    x = element.split()
    amenities_list.append(len(x))

data.amenities = amenities_list

data

# split the training and test sets with a 60% and 40% size of original
split_data = data.drop(['price'], axis = 1)

train1, test1, train2, test2 = cross_validation.train_test_split(split_data,
                                                data.price, 
                                                test_size=0.4,
                                                train_size = 0.6,
                                                random_state=13)

# mean of prices
mean = np.mean(data.price)

# standard deviation to compare 
std = np.std(data.price)

print("mean: " + str(mean))
print ("standard deviation: " + str(std))

# linear regression testing
linear_reg = linear_model.LinearRegression()
linear_reg.fit(train1, train2)
linear_reg_error = metrics.median_absolute_error(test2, linear_reg.predict(test1))

# ridge model testing
ridge = linear_model.Ridge()
ridge.fit(train1, train2)
ridge_error = metrics.median_absolute_error(test2, ridge.predict(test1))
print ("Linear Regression: " + str(linear_reg_error))
print ("Ridge: " + str(ridge_error))

# ada boost regressor
param_names = ["n_estimators", "learning_rate", "loss"]
param_values = [[1], [1,2], ['linear']]

parameters = dict(zip(param_names, param_values))

abr = GridSearchCV(ensemble.AdaBoostRegressor(),
                   cv = 3,
                   param_grid = parameters,
                   scoring = 'median_absolute_error')
preds = abr.fit(train1, train2)
abr_best_estimator = abr.best_estimator_

# gradient boosting regressor
param_names = ["n_estimators", "max_depth", "learning_rate", "loss", "min_samples_split"]
param_values = [[100,300,500], [1,2,3,4], [0.01, 0.02], ['ls', 'lad'], [1,2]]

parameters = dict(zip(param_names, param_values))
gbr = GridSearchCV(ensemble.GradientBoostingRegressor(), 
                   cv = 3, 
                   param_grid = parameters, 
                   scoring='median_absolute_error')
preds = gbr.fit(train1, train2)
gbr_best_estimator = gbr.best_estimator_

# bagging regressor
param_names = ["n_estimators", "max_features", "bootstrap", "oob_score"]
param_values = [[100,500,1000], [1,5,10], [True, False], [False]]

parameters = dict(zip(param_names, param_values))
br = GridSearchCV(ensemble.BaggingRegressor(), 
                   cv = 3, 
                   param_grid = parameters, 
                   scoring='median_absolute_error')
preds = br.fit(train1, train2)
br_best_estimator = br.best_estimator_

print("AdaBoostRegressor: " + "$" + str(abs(abr.best_score_)))
print("GradientBoostingRegressor: " + "$" + str(abs(gbr.best_score_)))
print("BaggingRegressor: " + "$" + str(abs(br.best_score_)))

gbr_best_estimator

# graph the features and importance
feature_importance = gbr_best_estimator.feature_importances_
feature_score = 100.0 * (feature_importance / feature_importance.max())

feature = pd.Series(feature_score, index = train1.columns).sort_values(ascending = True)

feature.plot(kind = 'barh',
              figsize = (15,10),
              color = 'LightBlue',
              title = 'Feature Importance')













