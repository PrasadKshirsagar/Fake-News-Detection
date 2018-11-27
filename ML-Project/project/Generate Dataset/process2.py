posInstances = []	#it holds the fake news
negInstances = []	#it holds the real news
from decimal import *

m = 2
k_attr = 100

lines = [line.rstrip('\n') for line in open('../Dataset1/dataset1.txt')]
totExam = len(lines)
D = len(lines[0].split(' '))-2
evidence = [Decimal(0)]*D


for line in lines:
	l = line.split(" ")
	if int(l[D])==0:
		d = [0]*D
		i = 0
		for item in l:
			if(i==D):
				break
			else:
				d[i] = int(item)
				evidence[i] = evidence[i] + Decimal(item)
				i = i + 1
		negInstances.append(d)
	elif int(l[D])==1:
		d = [0]*D
		i = 0
		for item in l:
			if(i==D):
				break
			else:
				d[i] = int(item)
				evidence[i] = evidence[i] + Decimal(item)
				i = i + 1
		posInstances.append(d)

totPos = len(posInstances)
totNeg = len(negInstances)
xis1givenNeg = [Decimal(0)]*D

for instance in negInstances:
	i = 0
	for item in instance:
		if(not item==0):
			xis1givenNeg[i] = xis1givenNeg[i] + Decimal(1)
		i = i + 1
del negInstances


xis1givenPos = [Decimal(0)]*D
for instance in posInstances:
	i = 0
	for item in instance:
		if(not item==0):
			xis1givenPos[i] = xis1givenPos[i] + Decimal(1)
		i = i + 1
del posInstances


priorPos = Decimal(totPos)/(totPos+totNeg)
priorNeg = Decimal(totNeg)/(totNeg+totPos)
print("prior prob of fake news: "+str(priorNeg))
print("prior prob of real news: "+str(priorPos))

for i in range(D):
	xis1givenPos[i] = (xis1givenPos[i]+1)/(totPos+m)
	xis1givenNeg[i] = (xis1givenNeg[i]+1)/(totNeg+m)

del lines

# NOT REQD AS totExam is same for all i
#for i in range(D):
#	evidence[i] = evidence[i]/totExam


posterNeg = [Decimal(0)]*D
tot0 = 0
for i in range(D):
	posterNeg[i] = (xis1givenNeg[i]*priorNeg)/((xis1givenNeg[i]*priorNeg)+(xis1givenPos[i]*priorPos))

attributesToKeep = {}


import numpy as np
arr = np.array(posterNeg)
topk = arr.argsort()[-k_attr:][::-1]
topk = list(topk)

bottomk = arr.argsort()[:k_attr]
bottomk = list(bottomk)

for item in topk:
	attributesToKeep[item] = 1
	print(str(item)+" "+str(posterNeg[item])+" ")

for item in bottomk:
	attributesToKeep[item] = 1
	print(str(item)+" "+str(posterNeg[item])+" ")

print(attributesToKeep)

f = open('../Dataset2/NEW_myX','w')
lines = [line.rstrip('\n') for line in open('../Dataset1/dataset1.txt')]
for line in lines:
	l = line.split(" ")
	i = 0
	while i<D:
		if i in attributesToKeep:
			val = ''
			if l[i]=='0':
				val = '0'
			else:
				val = '1'
			f.write(val+" ")
		i = i + 1
	f.write(l[D]+" \n")


f1 = open('../Dataset2/NEW_myX_test','w')
lines1 = [line.rstrip('\n') for line in open('../Dataset1/dataset2.txt')]
for line in lines1:
	l = line.split(" ")
	i = 0
	while i<D:
		if i in attributesToKeep:
			val = ''
			if l[i]=='0':
				val = '0'
			else:
				val = '1'
			f1.write(str(val)+" ")
		i = i + 1
	f1.write(l[D]+" \n")

