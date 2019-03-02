# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
y = dataset.iloc[:, -1].values
# Cleaning the texts
import re
# import nltk
# nltk.download('stopwords') package already download
from nltk.corpus import stopwords #irrelevant words (determiners, prepositions, etc ..)
irrelevantWords = set(stopwords.words('english'))
from nltk.stem.porter import PorterStemmer #Stemming Process (Taking care of tenses etc.)
ps = PorterStemmer()
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in irrelevantWords]
    review = ' '.join(review)
    # dataset.at[i, 'Review'] = review
    corpus.append(review)

# Creating Bag Of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #includes many preprocessing parameters lowercasing, stemming etc. 
x = cv.fit_transform(corpus).toarray()

# Splitting into TestSet and Training Set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Classifier on Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(xTrain, yTrain)

# Predict Test Set Results
yPred = classifier.predict(xTest)

# Make Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)
