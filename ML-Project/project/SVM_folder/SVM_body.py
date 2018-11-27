from __future__ import division
import pandas as pd
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from nltk.tokenize import word_tokenize 
from collections import OrderedDict
from sklearn.svm import SVC
import numpy as np


ps = PorterStemmer()
stop_words = set(stopwords.words('english')) 


# storing body & title of all instances in arrays
df = pd.read_csv('fake.csv')
df2 = pd.read_csv('articles1.csv')

saved_title = df.title
saved_body = df.text
saved_language = df.language


saved_true_title = df2.title
saved_true_body = df2.content


f = open('dataset', 'w')


print('Preprocessing data... ')

#---------------------------------# making frequency map :------------------------------------------------------------


# dict1 is a map containg frequency of all attributes

dict1 = {}

for i in range(0,len(saved_body)):
	if(saved_language[i] == 'english'):
		tmp = "'''"+str(saved_body[i])+"'''"
		tmp = tmp.replace('\n', ' ')
		words = word_tokenize(tmp.decode('utf-8'))
		filtered_sentence = [w1 for w1 in words if not w1 in stop_words]
		for w in filtered_sentence :
			if(w.isalnum()):
				temp = ps.stem(w)
				if(dict1.has_key(temp)):
					count = dict1[temp] + 1
					dict1[temp] = count
				else :
					dict1[temp] = 1	


it = 0
selected_features = []
# selected_features is al list containing list of selected 5000 features 

a1_sorted_keys = sorted(dict1, key=dict1.get, reverse=True)
for r in a1_sorted_keys:
    #print r, dict1[r]
    selected_features.append(r)
    it = it + 1
    if(it == 5000):
		break 



#---------------------------------# for training set X :------------------------------------------------------------

# X[12000][5001]    .....last column is label

N=12000
D = 5000
X = [[0 for x in range(D)] for y in range(N)]
Y = []

i=0
l=0
while (1):
	if(saved_language[l] == 'english'):
		tmp = "'''"+str(saved_body[l])+"'''"
		tmp = tmp.replace('\n', ' ')
		words = word_tokenize(tmp.decode('utf-8'))
		filtered_sentence = [w1 for w1 in words if not w1 in stop_words]
		us = set(filtered_sentence)

		# giving label
		Y.append(1)

		for k in range(0,5000):
			if(selected_features[k] in us):
				X[i][k] = 1
			else :
				X[i][k] = 0
		i = i+1
		if(i == 6000):
			break

	l = l + 1	



i = 0
while (i < 6000):
	tmp = "'''"+str(saved_true_body[i])+"'''"
	tmp = tmp.replace('\n', ' ')
	words = word_tokenize(tmp.decode('utf-8'))
	filtered_sentence = [w1 for w1 in words if not w1 in stop_words]
	us = set(filtered_sentence)
	Y.append(0) 
	for k in range(0,5000):
		if(selected_features[k] in us):
			X[i+6000][k] = 1
		else :
			X[i+6000][k] = 0
	i = i+1	


print('processed training file ... ')





#---------------------------------# for test set X_test :------------------------------------------------------------

# X_test[6000][5000]

N1=2000
D1 = 5000
X_test = [[0 for x in range(D1)] for y in range(N1)]
Y_test = []

i=0
l=0
while (1):
	if(saved_language[l+7000] == 'english'):
		tmp = "'''"+str(saved_body[l+7000])+"'''"
		tmp = tmp.replace('\n', ' ')
		words = word_tokenize(tmp.decode('utf-8'))
		filtered_sentence = [w1 for w1 in words if not w1 in stop_words]
		us = set(filtered_sentence)

		# giving label
		Y_test.append(1)

		for k in range(0,5000):
			if(selected_features[k] in us):
				X_test[i][k] = 1
			else :
				X_test[i][k] = 0
		i = i+1
		if(i == 1000):
			break

	l = l + 1	




i = 0
while (i < 1000):
	tmp = "'''"+str(saved_true_body[i+7000])+"'''"
	tmp = tmp.replace('\n', ' ')
	words = word_tokenize(tmp.decode('utf-8'))
	filtered_sentence = [w1 for w1 in words if not w1 in stop_words]
	us = set(filtered_sentence)
	Y_test.append(0)
	for k in range(0,5000):
		if(selected_features[k] in us):
			X_test[i+1000][k] = 1
		else :
			X_test[i+1000][k] = 0
	i = i+1	

print('processed testing file ... ')


def column(matrix,col):
    return [matrix[i][col] for i in range(len(matrix))]


#----------------------------------------------Applying SVM -----------------------------------------------------


print('training ...')
model = SVC().fit(X, Y)
print('predicting ...')
v = model.predict(X_test)

ct = 0
for i in range(0,len(v)):
	if(v[i] == Y_test[i]):
		ct = ct + 1

accuracy = 	ct * 100/len(v)
print('=====> Accuracy of svm model is : '+ str(accuracy))
confusion_matrix1 = confusion_matrix(Y_test, v)
print(confusion_matrix1)









					











