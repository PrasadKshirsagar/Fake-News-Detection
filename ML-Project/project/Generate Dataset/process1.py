import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import OrderedDict

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# storing body & title of all instances in arrays
df = pd.read_csv('../Dataset1/fake.csv')
df2 = pd.read_csv('../Dataset1/articles1.csv')

saved_title = df.title
saved_body = df.text
saved_language = df.language

saved_true_title = df2.title
saved_true_body = df2.content

f1 = open('../Dataset1/dataset1.txt', 'w')
f2 = open('../Dataset1/dataset2.txt', 'w')

# ---------------------------------# making frequency map :------------------------------------------------------------


# dict1 is a map containg frequency of all attributes

dict1 = {}
stttt = ""
for i in range(0, len(saved_body)):
    if (saved_language[i] == 'english'):
        tmp = "'''" + str(saved_body[i]) + "'''"
        tmp = tmp.replace('\n', ' ')
        words = word_tokenize(tmp)
        filtered_sentence = [w1 for w1 in words if not w1 in stop_words]
        for w in filtered_sentence:
            stttt = stttt+w
            if (w.isalnum()):
                temp = ps.stem(w)
                if (temp in dict1):
                    count = dict1[temp] + 1
                    dict1[temp] = count
                else:
                    dict1[temp] = 1

it = 0
selected_features = []
# selected_features is al list containing list of selected 5000 features 



a1_sorted_keys = sorted(dict1, key=dict1.get, reverse=True)
for r in a1_sorted_keys:
    print(r, dict1[r])
    selected_features.append(r)
    it = it + 1
    if (it == 5000):
        break

    # ---------------------------------# for training set X :------------------------------------------------------------

# X[12000][5001]    .....last column is label

N = 12000
D = 5001
X = [[0 for x in range(D)] for y in range(N)]

i = 0
l = 0
while (1):
    if (saved_language[l] == 'english'):
        tmp = "'''" + str(saved_body[l]) + "'''"
        tmp = tmp.replace('\n', ' ')
        words = word_tokenize(tmp)
        filtered_sentence = [w1 for w1 in words if not w1 in stop_words]
        us = set(filtered_sentence)

        # giving label
        X[i][5000] = 1

        for k in range(0, 5000):
            if (selected_features[k] in us):
                X[i][k] = 1
            else:
                X[i][k] = 0
        i = i + 1
        if (i == 6000):
            break

    l = l + 1

i = 0
while (i < 6000):
    tmp = "'''" + str(saved_true_body[i]) + "'''"
    tmp = tmp.replace('\n', ' ')
    words = word_tokenize(tmp)
    filtered_sentence = [w1 for w1 in words if not w1 in stop_words]
    us = set(filtered_sentence)
    X[i + 6000][5000] = 0
    for k in range(0, 5000):
        if (selected_features[k] in us):
            X[i + 6000][k] = 1
        else:
            X[i + 6000][k] = 0
    i = i + 1

# writing X into file
# for j in range(0, N):
#     for k in range(0, D):
#         print(str(X[j][k]) + " ")


for j in range(0,N):
	for k in range(0,D):
		f1.write(str(X[j][k])+" ")
	f1.write("\n")

f1.close()

# ---------------------------------# for test set X_test :------------------------------------------------------------

# X_test[6000][5000]

N1 = 6000
D1 = 5001
X_test = [[0 for x in range(D1)] for y in range(N1)]

i = 0
l = 0
while (1):
    if (saved_language[l + 7000] == 'english'):
        tmp = "'''" + str(saved_body[l + 7000]) + "'''"
        tmp = tmp.replace('\n', ' ')
        words = word_tokenize(tmp)
        filtered_sentence = [w1 for w1 in words if not w1 in stop_words]
        us = set(filtered_sentence)

        # giving label
        X_test[i][5000] = 1

        for k in range(0, 5000):
            if (selected_features[k] in us):
                X_test[i][k] = 1
            else:
                X_test[i][k] = 0
        i = i + 1
        if (i == 3000):
            break

    l = l + 1

i = 0
while (i < 3000):
    tmp = "'''" + str(saved_true_body[i + 7000]) + "'''"
    tmp = tmp.replace('\n', ' ')
    words = word_tokenize(tmp)
    filtered_sentence = [w1 for w1 in words if not w1 in stop_words]
    us = set(filtered_sentence)
    X_test[i + 3000][5000] = 0
    for k in range(0, 5000):
        if (selected_features[k] in us):
            X_test[i + 3000][k] = 1
        else:
            X_test[i + 3000][k] = 0
    i = i + 1

# output X & X_test to check
for j in range(0,N1):
    for k in range(0,D1):
        f2.write(str(X_test[j][k])+" ")
    f2.write("\n")

f2.close()
