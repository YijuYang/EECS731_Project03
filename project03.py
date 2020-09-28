# compare algorithms
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import cluster

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load dataset
names = ['movieId','imdbId','tmdbId']
dataset_links = read_csv('./ml-latest-small/links.csv', names=names)

names = ['movieId','title','genres']
dataset_movies = read_csv('./ml-latest-small/movies.csv', names=names)

names = ['userId','movieId','rating','timestamp']
dataset_ratings = read_csv('./ml-latest-small/ratings.csv', names=names)

names = ['userId','movieId','tag','timestamp']
dataset_tags = read_csv('./ml-latest-small/tags.csv', names=names)

dataset_links   = dataset_links.dropna()
dataset_movies  = dataset_movies.dropna()
dataset_ratings = dataset_ratings.dropna()
dataset_tags    = dataset_tags.dropna()

# dataset_tags = dataset_tags.sort_values('movieId')
dataset_tags = dataset_tags.drop(['userId','timestamp'], axis=1)
dataset_ratings = dataset_ratings.drop(['userId','timestamp'], axis=1)
dataset_movies.sort_values(by=['movieId'])
print(dataset_ratings)
pyplot.plot(dataset_ratings['movieId'],dataset_ratings['rating'],'o')
pyplot.ylabel('rating')
pyplot.xlabel('movieId')
pyplot.axis([0, 10.5, 0, 5.5])
pyplot.show()

print(dataset_tags)
label_encoder = preprocessing.LabelEncoder()
pyplot.plot(dataset_tags['movieId'],label_encoder.fit_transform(dataset_tags['tag']),'o')
pyplot.ylabel('tage')
pyplot.xlabel('movieId')
pyplot.axis([0, 1000, 0, 2000])
pyplot.show()

model_tm = cluster.KMeans(n_clusters = 10)
predict_tm = model_tm.fit(label_encoder.fit_transform(dataset_tags['tag']).reshape(-1, 1),dataset_tags['movieId'])
labels_tm = predict_tm.labels_
print(labels_tm)

#Find average rating for all movies
movies = []
avg_rating = []
for currMovie in set(dataset_ratings['movieId']):
    list = [i for i,x in enumerate(dataset_ratings['movieId']) if x==currMovie]
    count = 0
    total = 0
#     print(len(list))
    for k in list:
        count += 1
        total += dataset_ratings['rating'][k]
    if count != 0:
        movies.append(currMovie)
        avg_rating.append(total/count)

pyplot.plot(movies,avg_rating,'o')
pyplot.ylabel('avg_rating')
pyplot.xlabel('movieId')
pyplot.axis([0, 10000, 0, 5.5])
pyplot.show()

model_rm = cluster.KMeans(n_clusters = 10)
predict_rm = model_rm.fit(np.array(avg_rating).reshape(-1, 1),movies)
labels_rm = predict_rm.labels_
print(labels_rm)
recommendation = []
for x in range(min(len(labels_rm),len(labels_tm))):
    if labels_rm[x] == labels_tm[x]:
        recommendation.append((movies[x], True))
    else:
        recommendation.append((movies[x], False))
print(recommendation)

# r1 = np.concatenate((data_loader2['age'], data_loader3['age']))
# r2 = np.concatenate((data_loader2['charges'], data_loader3['policy_annual_premium']))
# r3 = np.concatenate((data_loader2['region'], data_loader3['policy_state']))
# newdata = pd.DataFrame({'age':r1,'charges':r2,'region':r3})
#
