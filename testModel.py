# test naive bayes and linear SVM classifiers on the 2005 Congress debate transcripts

import numpy as np
import os, fnmatch
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import text 

def findFiles (path, filter):
	for root, dirs, files in os.walk(path):
		for file in fnmatch.filter(files, filter):
			yield os.path.join(root, file)

speechArray = []
partyArray = []

democratCount = 0
republicanCount = 0

for textFile in findFiles(r'*** CONGRESS SPEECH FOLDER ***', '*.txt'):
	with open(textFile, 'r') as speechFile:
		speech = speechFile.read()
		speechArray.append(speech)

		# check whether the speech is from a republican or a democrat from the filename
		if str(textFile)[-7] == 'D':
			partyArray.append('Democrat')
			democratCount += 1
		else:
			partyArray.append('Republican')
			republicanCount += 1

print "Number of Democrat speeches: "+str(democratCount)
print "Number of Republican speeches: "+str(republicanCount)
print

X = np.array(speechArray)
y = np.array(partyArray)
skf = StratifiedKFold(y, n_folds=10, shuffle=True)

totalF1SumNB = 0
democratF1SumNB = 0
republicanF1SumNB = 0
totalF1SumSVM = 0
democratF1SumSVM = 0
republicanF1SumSVM = 0

# 10-fold cross-validation
for train_index, test_index in skf:
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	count_vect = CountVectorizer(stop_words='english', ngram_range=(1,2))
	X_train_counts = count_vect.fit_transform(X_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	# Naive Bayes
	clf = MultinomialNB().fit(X_train_tfidf, y_train)

	X_new_counts = count_vect.transform(X_test)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	predicted = clf.predict(X_new_tfidf)
	print "Naive Bayes:"
	print "mean: "+str(np.mean(predicted == y_test))
	print metrics.classification_report(y_test, predicted)
	print "f1_score: "+str(f1_score(y_test, predicted, average=None))
	democratF1SumNB += f1_score(y_test, predicted, average=None)[0]
	republicanF1SumNB += f1_score(y_test, predicted, average=None)[1]
	totalF1SumNB += (f1_score(y_test, predicted, average=None)[0]+f1_score(y_test, predicted, average=None)[1])/2
	print
	print

	# SVM
	clf = SGDClassifier().fit(X_train_tfidf, y_train)

	X_new_counts = count_vect.transform(X_test)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	predicted = clf.predict(X_new_tfidf)
	print "SVM:"
	print "mean: "+str(np.mean(predicted == y_test))
	print metrics.classification_report(y_test, predicted)
	print "f1_score: "+str(f1_score(y_test, predicted, average=None))
	democratF1SumSVM += f1_score(y_test, predicted, average=None)[0]
	republicanF1SumSVM += f1_score(y_test, predicted, average=None)[1]
	totalF1SumSVM += (f1_score(y_test, predicted, average=None)[0]+f1_score(y_test, predicted, average=None)[1])/2
	print
	print

print "Naive Bayes:"
print "Democrat average f1_score: "+str(democratF1SumNB/10)
print "Republican average f1_score: "+str(republicanF1SumNB/10)
print "Total average f1_score: "+str(totalF1SumNB/10)
print
print "SVM:"
print "Democrat average f1_score: "+str(democratF1SumSVM/10)
print "Republican average f1_score: "+str(republicanF1SumSVM/10)
print "Total average f1_score: "+str(totalF1SumSVM/10)
