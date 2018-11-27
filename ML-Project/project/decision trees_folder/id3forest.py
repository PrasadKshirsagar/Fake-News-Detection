import sys
sys.setrecursionlimit(10**6)
D = 200
numtrees = 15
attrPerTree = 120

class node:
    def __init__(self, attrib, posinst, neginst, label):	
        self.left = None 				#left child in the tree
        self.right = None
        self.attrib = attrib 			#attrib is the index of the word used as attribute in this node
        self.posinst = posinst 			#posinst is the number of positive examples at this node
        self.neginst = neginst
        self.label = label				#label = "notleaf" OR "yes" if the prediction is positive(i.e. the movie review is positive) OR "no"-if the prediction is negative

    def insert(self, attrib, posinst, neginst, lorR, label):
    	if lorR == 'left':				#insert at the left child of the node
    		self.left = node(attrib, posinst, neginst, label)
    		return self.left
    	elif lorR == 'right':
    		self.right = node(attrib, posinst, neginst, label)
    		return self.right

import math
def entropy(numpos,numneg):		#caclulate the entropy where numpos is the number of positive examples
	total = numpos + numneg
	if total==0:
		return 0
	ppos = numpos/total
	pneg = numneg/total
	if(ppos==0 or pneg==0):
		return 0
	return (-ppos)*math.log2(ppos)-pneg*math.log2(pneg)

def splitting(node1,posI,negI,attr,parent):			#to create the children of node1 and then this is recursively called
	#print(len(posI),len(negI))
	
		ig = {}						#ig keeps the info gain for each of the attributes
		for index in attr:			#one by one see all attributes to check which has the max info gain
			yespos = 0				#num of positive examples(review pos) who has this "index" attribute
			nopos = 0				#num of positive examples who does not have this attribute
			yesneg = 0
			noneg = 0
			for itemDict in posI:		#posI is a list of dictionaries with each dictionary containing one positive review from the training set
				if int(index) in itemDict:
					yespos = yespos + 1
				else:
					nopos = nopos + 1
			for itemDict in negI:
				if int(index) in itemDict:
					yesneg = yesneg + 1
				else:
					noneg = noneg + 1
			totalI = len(posI) + len(negI)
			ig[index] = entropy(len(posI),len(negI)) - (yespos+yesneg)/totalI*entropy(yespos,yesneg) - (nopos+noneg)/totalI*entropy(nopos,noneg)
			#print(nopos, noneg, entropy(nopos,noneg))
		maxig = ig[list(ig.keys())[0]]			#find the max info gain of all the ig
		indexOfmaxig = list(ig.keys())[0]		#stores the index of max info gain
		for key in ig:
			if(ig[key]>maxig):
				maxig = ig[key]
				indexOfmaxig = key
		if maxig==0:
			if len(posI)<len(negI):
				node1.label = "no"
			else:
				node1.label = "yes"
		else:
			node1.attrib = indexOfmaxig			#made the attribute of this node as the one with max ig
			yesposI = []		#list of all instances which are positive reviewed and contains the attribute "indexofmaxig"
			yesnegI = []
			noposI = []
			nonegI = []
			for dictionary in posI:
				if int(indexOfmaxig) in dictionary:
					yesposI.append(dictionary)
				else:
					noposI.append(dictionary)
			for dictionary in  negI:
				if int(indexOfmaxig) in dictionary:
					yesnegI.append(dictionary)
				else:
					nonegI.append(dictionary)
			if len(yesposI)+len(yesnegI)==0:
				if len(posI)>len(negI):
					node1.label = "yes"
				else:
					node1.label = "no"
			elif len(noposI)+len(nonegI)==0:
				if len(posI)>len(negI):
					node1.label = "yes"
				else:
					node1.label = "no"
			else:
				node2 = node1.insert("attr",yesposI,yesnegI,"left","notleaf")	#create the two children of node1
				node3 = node1.insert("attr",noposI,nonegI,"right","notleaf")
				attrr = []
				attrri = []
				for item in attr:
					attrr.append(item)
					attrri.append(item)
				attrr.remove(indexOfmaxig)		#remove the attribute and then use the new attribute list
				attrri.remove(indexOfmaxig)
				splitting(node2,yesposI,yesnegI,attrr,node1)
				splitting(node3,noposI,nonegI,attrr,node1)

import random
attributesMain = list(range(D))
rootList = []
lines = [line.rstrip('\n') for line in open('NEW_myX')]
for loop in  range(1,numtrees+1):
	attributes = []
	attrmap = {};
	itr = 0
	for item1 in attributesMain:
		attrmap[itr] = item1
		itr = itr + 1
	lis = random.sample(range(0,D-1),attrPerTree)
	for item1 in lis:
		attributes.append(attrmap[item1])
	posInstances = []
	negInstances = []
	for line in lines:
		l = line.split(" ")
		if int(l[D])==0:
			d = {}
			i = 0
			for item in l:
				if(i==D):
					break
				else:
					if(int(item)!=0):
						d[i] = int(item)
					i = i + 1
			negInstances.append(d)
		elif int(l[D])==1:
			d = {}
			i = 0
			for item in l:
				if(i==D):
					break
				else:
					if(int(item)!=0):
						d[i] = int(item)
					i = i + 1
			posInstances.append(d)

	ig = {}
	for index in attributes:
		yespos = 0
		nopos = 0
		yesneg = 0
		noneg = 0
		for itemDict in posInstances:
			if int(index) in itemDict:
				yespos = yespos + 1
			else:
				nopos = nopos + 1
		for itemDict in negInstances:
			if int(index) in itemDict:
				yesneg = yesneg + 1
			else:
				noneg = noneg + 1
		totalI = len(posInstances) + len(negInstances)
		ig[index] = entropy(len(posInstances),len(negInstances)) - (yespos+yesneg)/totalI*entropy(yespos,yesneg) - (nopos+noneg)/totalI*entropy(nopos,noneg)
		#print(nopos, noneg, entropy(nopos,noneg))
	maxig = ig[list(ig.keys())[0]]
	indexOfmaxig = list(ig.keys())[0]
	for key in ig:
		if(ig[key]>maxig):
			maxig = ig[key]
			indexOfmaxig = key
	#print(maxig)
	root = node(indexOfmaxig,posInstances,negInstances,"notleaf")
	temp = root
	if(len(posInstances)==0):
		root.label = "no"
	elif len(negInstances)==0:
		root.label = "yes"
	else:
		yesposI = []
		yesnegI = []
		noposI = []
		nonegI = []
		for dictionary in posInstances:
			if int(indexOfmaxig) in dictionary:
				yesposI.append(dictionary)
			else:
				noposI.append(dictionary)
		for dictionary in  negInstances:
			if int(indexOfmaxig) in dictionary:
				yesnegI.append(dictionary)
			else:
				nonegI.append(dictionary)
		node1 = root.insert("attr",yesposI,yesnegI,"left","notleaf")
		node2 = root.insert("attr",noposI,nonegI,"right","notleaf")
		attributes.remove(indexOfmaxig)
		splitting(node1,yesposI,yesnegI,attributes,root)
		splitting(node2,noposI,nonegI,attributes,root)
	rootList.append(root)
	print(str(loop)+"tree built")
#checking the test examples to find the percentage of correct predictions
totalcorrect = 0
del lines
lines = [line.rstrip('\n') for line in open('NEW_myX_test')]
total = len(lines)
for line in lines:
	l = line.split(" ")
	rating = int(l[D])
	ans = ""
	if(rating==1):
		ans = "fake"
	if(rating==0):
		ans = "real"
	myans = ""
	d = {}
	i = 0
	for item in l:
		if(i==D):
			break
		else:
			if(int(item)!=0):
				d[i] = int(item)
			i = i + 1

	numyes = 0
	numno = 0
	for itr in range(0,numtrees-1):
		temp = rootList[itr]
		while True:
			if temp.label=="notleaf":
				if int(temp.attrib) in d:
					temp = temp.left
				else:
					temp = temp.right
			else:
				if temp.label=="yes":
					numyes = numyes + 1
				elif temp.label=="no":
					numno = numno + 1
				break
	if(numyes>=numno):
		myans = "fake"
	else:
		myans = "real"
	if(myans==ans):
		totalcorrect = totalcorrect + 1

print(totalcorrect)
correctPercent = totalcorrect/total
correctPercent = correctPercent*100
print(correctPercent)

