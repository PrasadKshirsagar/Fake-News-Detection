import sys
sys.setrecursionlimit(10**6)
D = 200


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
    
    def count_attr_freq(self):
    	attr_freq = {}
    	self.count_attr_freq_helper(attr_freq)
    	return attr_freq

    def count_attr_freq_helper(self, attr_freq):
    	if self.label != "notleaf":
    		return
    	else:
    		if self.attrib in attr_freq:
    			attr_freq[self.attrib] = attr_freq[self.attrib] + 1
    		else:
    			attr_freq[self.attrib] = 1
    		if self.left:
    			self.left.count_attr_freq_helper(attr_freq)
    		if self.right:
    			self.right.count_attr_freq_helper(attr_freq)

    def count_leaf_nodes(self):
    	leaf_nodes = []
    	self.count_leaf_nodes_helper(leaf_nodes)
    	return len(leaf_nodes)

    def count_leaf_nodes_helper(self, leaf_nodes):
    	if self.label != "notleaf":
    		leaf_nodes.append(self)
    	else:
    		if self.left:
    			self.left.count_leaf_nodes_helper(leaf_nodes)
    		if self.right:
    			self.right.count_leaf_nodes_helper(leaf_nodes)

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
				if index in itemDict:
					yespos = yespos + 1
				else:
					nopos = nopos + 1
			for itemDict in negI:
				if index in itemDict:
					yesneg = yesneg + 1
				else:
					noneg = noneg + 1
			totalI = len(posI) + len(negI)
			if(totalI==0):
				ig[index] = 0
			else:
				ig[index] = entropy(len(posI),len(negI)) - ((yespos+yesneg)/totalI)*entropy(yespos,yesneg) - ((nopos+noneg)/totalI)*entropy(nopos,noneg)
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


attributes = list(range(D))		#0 to 4999
posInstances = []
negInstances = []
lines = [line.rstrip('\n') for line in open('NEW_myX')]
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

del lines
ig = {}
for index in attributes:
	yespos = 0
	nopos = 0
	yesneg = 0
	noneg = 0
	for itemDict in posInstances:
		if index in itemDict:
			yespos = yespos + 1
		else:
			nopos = nopos + 1
	for itemDict in negInstances:
		if index in itemDict:
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
		if indexOfmaxig in dictionary:
			yesposI.append(dictionary)
		else:
			noposI.append(dictionary)
	for dictionary in negInstances:
		if indexOfmaxig in dictionary:
			yesnegI.append(dictionary)
		else:
			nonegI.append(dictionary)
	node1 = root.insert("attr",yesposI,yesnegI,"left","notleaf")
	node2 = root.insert("attr",noposI,nonegI,"right","notleaf")
	attributes.remove(indexOfmaxig)
	splitting(node1,yesposI,yesnegI,attributes,root)
	splitting(node2,noposI,nonegI,attributes,root)


#checking the test examples to find the percentage of correct predictions
totalcorrect = 0
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
	temp = root
	while True:
		if temp.label=="notleaf":
			if int(temp.attrib) in d:
				temp = temp.left
			else:
				temp = temp.right
		else:
			if temp.label=="yes":
				myans = "fake"
			elif temp.label=="no":
				myans = "real"
			if myans==ans:
				totalcorrect = totalcorrect + 1
			break
print(totalcorrect)
correctPercent = totalcorrect/total
correctPercent = correctPercent*100
print(correctPercent)
print("id3 without early stopping: \n\nAccuracy: ",correctPercent)
print("\nnum of leaves: ",root.count_leaf_nodes())
dict_attr_freq = root.count_attr_freq()
i = 0
print("\n5 most used attributes to split are(attributeIndex numOfTimesUsed): ")
for key in sorted(dict_attr_freq.items(), key=lambda x: x[1], reverse=True):
	i = i+1
	print(key[0], "\t", key[1])
	if i>4:
		break