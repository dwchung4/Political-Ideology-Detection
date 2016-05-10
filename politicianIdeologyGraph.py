# use linear SVM classifier to plot the probability distribution of a candidate’s speeches.
# Corresponds to Figure 1 - 7 in the report.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import text
import numpy as np
import os, fnmatch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py

py.sign_in('dcfive', '*** PASSWORD ***')

def findFiles (path, filter):
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file)

X_train = []
y_train = []

democratCount = 0
republicanCount = 0

for textFile in findFiles(r'*** CONGRESS SPEECH FOLDER ***', '*.txt'):
    # filter out speeches from presidential candidates
    # #300008: Biden, 300022: Clinton, #300039: Edwards, #400629: Obama, #412216: Bachmann, #300071: McCain, #400311: Paul, #300085: Santorum, #300158: Thompson
    if '300008' not in textFile and '300022' not in textFile and '300039' not in textFile and '400629' not in textFile and '412216' not in textFile and '300071' not in textFile and '400311' not in textFile and '300085' not in textFile and '300158' not in textFile:
        if os.path.getsize(textFile) > 100:
            with open(textFile, 'r') as speechFile:
                speech = speechFile.read()
                X_train.append(speech)

                # check whether the speech is from a republican or a democrat from the filename
                if str(textFile)[-7] == 'D':
                    y_train.append('Democrat')
                    democratCount += 1
                else:
                    y_train.append('Republican')
                    republicanCount += 1

X_test = []

for textFile in findFiles(r'*** CANDIDATE\'S ELECTION SPEECH FOLDER ***', '*.txt'):
    with open(textFile, 'r') as speechFile:
        speech = speechFile.read()
        X_test.append(speech)

# SVM

additional_stop_words = ['mr', 'speaker', 'people', 'time', 'chairman', 'gentleman', 'committee', 'minutes', 'seconds', 'amp', 'nbsp', 'lt', 'gt', 'br']

democratProbabilityList = []

for i in range(0, 10):

    count_vect = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(additional_stop_words), ngram_range=(1,3))
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = SGDClassifier(loss='log').fit(X_train_tfidf, y_train)
    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    probabilityListSVM = clf.predict_proba(X_new_tfidf)

    if i == 0:
        for j in range(0, len(probabilityListSVM)):
            democratProbabilityList.append(probabilityListSVM[j][0])
    else:
        for j in range(0, len(probabilityListSVM)):
            democratProbabilityList[j] += probabilityListSVM[j][0]

for i in range(0, len(democratProbabilityList)):
    democratProbabilityList[i] /= 10

print democratProbabilityList

numpy_hist = plt.figure()

plt.hist(democratProbabilityList, bins=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])

plot_url = py.plot_mpl(numpy_hist, filename='numpy-bins')

print np.mean(democratProbabilityList)

predicted = clf.predict(X_new_tfidf)

democratCount = 0
republicanCount = 0
for prediction in predicted:
    if prediction == 'Democrat':
        democratCount += 1
    else:
        republicanCount += 1
print 'Democrat/Republican: '+str(democratCount)+'/'+str(republicanCount)
print
print "SVM:"
print "mean: "+str(np.mean(predicted == y_test))
print metrics.classification_report(y_test, predicted)
print "f1_score: "+str(f1_score(y_test, predicted, average=None))