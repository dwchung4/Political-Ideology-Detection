Political Ideology Detection Using Text Classification

Measure politicians' ideologies using text classification to measure how Democratic or Republican their speeches are.
Test the hypothesis that candidates move towards the center of the political spectrum for general elections in order
to win more votes from the voters on the opposite side as well.

1. The 2005 US Congress debate transcripts can be downloaded at http://www.cs.cornell.edu/home/llee/data/convote.html

2. The 2008 and 2012 presidential election speeches can be downloaded at http://www.cs.cmu.edu/~ark/CLIP/index.html

3. testModel.py tests naive bayes and linear SVM classifiers and corresponds to Table 2 and Table 3 in the report.

4. testModelFilter.py tests naive bayes and linear SVM classifiers with additional filters and corresponds to Table 4 and Table 5
   in the report.

5. politicianIdeology.py: use linear SVM classifier to classify a politician’s speeches into Democratic and Republican.
   Corresponds to Table 6 and Table 7 in the report.

6. politicianIdeologyGraph.py: use linear SVM classifier to plot the probability distribution of a candidate’s election speeches.
   Corresponds to Figure 1 - 7 in the report.

7. politicianTfidf.py: get tf-idf of a politician from the speeches for primary/general election
