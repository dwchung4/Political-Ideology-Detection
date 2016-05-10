# get tf-idf of a politician from the speeches for primary/general election

from sklearn.feature_extraction.text import TfidfVectorizer
import os, fnmatch
import numpy as np

np.set_printoptions(threshold=np.inf)

def findFiles (path, filter):
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file)

def getKey(item):
	return item[0]

corpus = []
speech = ''

abbrevs = [("'ve", " have"), ("i'm", "i am"), ("they're", "they are"), ("can't", "can not"), ("n't", " not"), ("'ll", " will"),
	("that's", "that is"), ("what's", "what is")]
ignoreList = [',', '.', ';', ':', '?', '\"']

folderList = [x[0] for x in os.walk('*** CONGRESS SPEECH FOLDER')]
for folder in folderList:
	if '*** FOLDER OF CANDIDATE\'S SPEECHES ***' in str(folder):
		for textFile in findFiles(folder, '*.txt'):
			with open(textFile, 'r') as speechFile:
				speech = speechFile.read()
				for abbrev in abbrevs:
					speech = speech.replace(abbrev[0], abbrev[1])
				for ignore in ignoreList:
					speech = speech.replace(ignore, '')
				speech += speech

corpus.append(speech)

for folder in folderList:
	if '*** FOLDER OF CANDIDATE\'S SPEECHES ***' not in str(folder):
		otherSpeech = ''
		for textFile in findFiles(folder, '*.txt'):
			with open(textFile, 'r') as speechFile:
				speech = speechFile.read()
				for abbrev in abbrevs:
					speech = speech.replace(abbrev[0], abbrev[1])
				otherSpeech += speech
	corpus.append(otherSpeech)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2,3), min_df=10)
X = vectorizer.fit_transform(corpus).toarray()
myList = X[0]
wordList = vectorizer.get_feature_names()

myTuple = []

for i in range(0, len(myList)):
	myTuple.append((myList[i], wordList[i]))

myTuple = sorted(myTuple, key=getKey, reverse=True)

for i in range(0, len(myTuple)):
	tfidf = str(myTuple[i][0])
	word = myTuple[i][1].encode('utf8')
	print "General: "+"("+str(tfidf)+", "+word+")"