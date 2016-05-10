# Use linear SVM classifier to classify a politician's speeches into
# Democratic and Republican

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
		# filter out short speeches
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

print "Number of Democrat speeches: "+str(democratCount)
print "Number of Republican speeches: "+str(republicanCount)
print

X_test = []
y_test = []

# candidate's speeches to test
for textFile in findFiles(r'*** CANDIDATE\'S SPEECH FOLDER ***', '*.txt'):
	with open(textFile, 'r') as speechFile:
		speech = speechFile.read()
		X_test.append(speech)
		y_test.append('Democrat') # 'Republican' for a Republican candidate

print "Number of candidate's speeches: "+str(len(X_test))
print

# SVM

additional_stop_words = ['mr', 'speaker', 'people', 'time', 'chairman', 'gentleman', 'committee', 'minutes', 'seconds', 'amp', 'nbsp', 'lt', 'gt', 'br']

democratCount = 0
republicanCount = 0

for i in range(0, 10):

	count_vect = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(additional_stop_words), ngram_range=(1,3))
	X_train_counts = count_vect.fit_transform(X_train)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	clf = SGDClassifier(loss='log').fit(X_train_tfidf, y_train)
	X_new_counts = count_vect.transform(X_test)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	predicted = clf.predict(X_new_tfidf)

	for prediction in predicted:
		if prediction == 'Democrat':
			democratCount += 1
		else:
			republicanCount += 1

print democratCount
print republicanCount