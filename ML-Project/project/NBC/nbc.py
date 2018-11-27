posInstances = []	#it holds the fake news
negInstances = []	#it holds the real news
from decimal import *

D = 200
m = 0

lines = [line.rstrip('\n') for line in open('NEW_myX')]
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
				i = i + 1
		posInstances.append(d)

totPos = len(posInstances)
totNeg = len(negInstances)
xis0givenNeg = [Decimal(0)]*D
xis1givenNeg = [Decimal(0)]*D

for instance in negInstances:
	i = 0
	for item in instance:
		if(item==0):
			xis0givenNeg[i] = xis0givenNeg[i] + Decimal(1)
		else:
			xis1givenNeg[i] = xis1givenNeg[i] + Decimal(1)
		i = i + 1
del negInstances
xis0givenPos = [Decimal(0)]*D
xis1givenPos = [Decimal(0)]*D

for instance in posInstances:
	i = 0
	for item in instance:
		if(item==0):
			xis0givenPos[i] = xis0givenPos[i] + Decimal(1)
		else:
			xis1givenPos[i] = xis1givenPos[i] + Decimal(1)
		i = i + 1
del posInstances
priorPos = Decimal(totPos)/(totPos+totNeg)
priorNeg = Decimal(totNeg)/(totNeg+totPos)

for i in range(D):
	xis1givenPos[i] = (xis1givenPos[i]+1)/(totPos+m)
	xis0givenPos[i] = (xis0givenPos[i]+1)/(totPos+m)
	xis1givenNeg[i] = (xis1givenNeg[i]+1)/(totNeg+m)
	xis0givenNeg[i] = (xis0givenNeg[i]+1)/(totNeg+m)

del lines

totalcorrect = 0
lines = [line.rstrip('\n') for line in open('NEW_myX_test')]
total = len(lines)
for line in lines:
	l = line.split(" ")
	ans = ""
	myans = ""
	d = [0]*D
	if int(l[D])==0:
		ans = "real"
		i = 0
		for item in l:
			if(i==D):
				break
			else:
				d[i] = int(item)
				i = i + 1
	elif int(l[D])==1:
		ans = "fake"
		i = 0
		for item in l:
			if(i==D):
				break
			else:
				d[i] = int(item)
				i = i + 1

	Pfake = Decimal(priorPos)
	Preal = Decimal(priorNeg)
	i = 0
	for item in d:
		if(item==0):
			Pfake = Pfake*xis0givenPos[i]
			Preal = Preal*xis0givenNeg[i]
		else:
			Pfake = Pfake*xis1givenPos[i]
			Preal = Preal*xis1givenNeg[i]
		i = i + 1

	if(Pfake>Preal):
		myans = "fake"
	else:
		myans = "real"

	if(myans==ans):
		totalcorrect = totalcorrect + 1

print(totalcorrect)
correctPercent = totalcorrect/total
correctPercent = correctPercent*100
print(correctPercent)