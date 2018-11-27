posInstances = []	#it holds the fake news
negInstances = []	#it holds the real news
from decimal import *

D = 5000
m = 0

lines = [line.rstrip('\n') for line in open('myXtitle')]
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
print("prior prob of fake news: "+str(priorNeg))
print("prior prob of real news: "+str(priorPos))

for i in range(D):
	xis1givenPos[i] = (xis1givenPos[i]+1)/(totPos+m)
	xis0givenPos[i] = (xis0givenPos[i]+1)/(totPos+m)
	xis1givenNeg[i] = (xis1givenNeg[i]+1)/(totNeg+m)
	xis0givenNeg[i] = (xis0givenNeg[i]+1)/(totNeg+m)

del lines

confmat = [0]*4			#0->fake fake 1->should be fake, is real 2->should be real, is fake  3->real real
totalcorrect = 0
lines = [line.rstrip('\n') for line in open('myX_testtitle')]
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
		if myans=="fake":
			confmat[0] = confmat[0]+1
		else:
			confmat[3] = confmat[3]+1
	else:
		if myans=="fake":
			confmat[2] = confmat[2]+1
		else:
			confmat[1] = confmat[1]+1

correctPercent = totalcorrect/total
correctPercent = correctPercent*100
print('Accuracy: ')
print(correctPercent)

print("\n Conf mat:")
print("\t fake\treal")
print("fake   "+str(confmat[2])+"\t"+str(confmat[3]))
print("real   "+str(confmat[0])+"\t"+str(confmat[1]))