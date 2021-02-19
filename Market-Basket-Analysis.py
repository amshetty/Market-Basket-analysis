import os
import sys
import json
import time
import pyspark
from operator import add
from pprint import pprint
from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict

ticTime = time.time()

## Initialising the SparkContext.
sc = SparkContext('local[*]', 'Task2')
sc.setLogLevel("ERROR")

## Defining the filter threshold.
filterThreshold = int(sys.argv[1])

## Defining the support threshold.
supportThreshold = int(sys.argv[2])

## Defining the path to the data.
dataSetPath = sys.argv[3]

## Output file path.
outfilePath = sys.argv[4]

## Loading the raw data.
rawData = sc.textFile(dataSetPath).map(lambda x: x.split(","))

## Filter out lines.
header = rawData.first()

## Remove the header.
rawData = rawData.filter(lambda line : line != header)

## Loading the data and preprocessing it.
reviewsRDD = rawData.map(lambda f : (f[0][1:-1][:-4] + f[0][1:-1][-2:] + '-' + str(int(f[1][1:-1])), str(int(f[5][1:-1])))).groupByKey().mapValues(list).map(lambda f: list(f[1])).filter(lambda f : len(f) > filterThreshold).map(set).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

## Function for implementing A-Priori algorithm.
def aPriori(chunkIterator, supportThreshold, numPartitions):

	## Scale down the support threshold.
	scaledDownSupportThreshold = supportThreshold // numPartitions

	## Accessing the baskets.
	chunkBaskets = list(chunkIterator)

	## Initialising a list to hold all the baskets.
	allBaskets = []

	## Initialise a global list to hold all the true frequent itemsets.
	trueFrequentGlobal = []

	## Frequency map for keeping a count of the candidate items.
	freqMap = defaultdict(int)

	## Process the singletons seperately.
	for basket in chunkBaskets:

		## Add the current basket to the list of baskets.
		allBaskets.append(basket)

		for candidateItem in basket:
			freqMap[candidateItem] += 1

	## Create a list of initial single frequent items.
	trueFrequents = []

	## Identify the true frequent items.
	for key, value in freqMap.items():

		## Check of count is greater than threshold.
		if (value >= scaledDownSupportThreshold):
			trueFrequents.append(set([key]))

	## Initialising the size of the candidate items.
	tupleSize = 2

	## Add the list to a global list.
	trueFrequentGlobal += trueFrequents

	## Termination Criteria.
	while (len(trueFrequents) > 0):

		## Obtain the next set of frequent items.
		trueFrequents = nextFrequentCandidates(trueFrequents, tupleSize, allBaskets, scaledDownSupportThreshold)

		## Add the list to a global list.
		trueFrequentGlobal += trueFrequents

		## Update the tuple size.
		tupleSize += 1

	return trueFrequentGlobal


## Function for computing the next set of frequent items.
def nextFrequentCandidates(trueFrequents, K, allBaskets, scaledDownSupportThreshold):

	## Generate all the possible candidates of size K.
	candidateItems = set()
	allPossibleCombinations = combinations(trueFrequents, 2)

	## Iterating over each combination.
	for newCombination in allPossibleCombinations:

		## Generate tuple.
		newCombination = newCombination[0].union(newCombination[1])

		# Add to the combination list if valid.
		if (len(newCombination) == K):
			candidateItems.add(frozenset(newCombination))

	## Frequency map for keeping a count of the candidate items.
	freqMap = defaultdict(int)

	## Initialise a set for holding the true frequent items.
	trueFrequent = []

	## Counting the number of occurences of the candidate items.
	for candidateItem in candidateItems:

		## Loop over all baskets.
		for basket in allBaskets:

			## Check whether this candidate is present or not.
			if (set(candidateItem).issubset(basket)):

				freqMap[candidateItem] += 1

				if (freqMap[candidateItem] == scaledDownSupportThreshold):
					trueFrequent.append(candidateItem)		
					break

	return trueFrequent

## Function for counting the true frequent items in the entire dataset.
def generateTrueFrequents(basketIterator, candidateList):

	## Accessing the baskets.
	basketList = list(basketIterator)

	## Frequency map for keeping a count of the candidate items.
	freqMap = defaultdict(int)

	## Counting the number of occurences of the candidate items.
	for candidateItem in candidateList:

		## Loop over all baskets.
		for basket in basketList:

			## Check whether this candidate is present or not.
			if ((set(candidateItem)).issubset(set(basket))):
					freqMap[candidateItem] += 1

	## Initialise a set for holding the true frequent items.
	trueFrequent = []	

	## Identify the true frequent items.
	for key, value in freqMap.items():
			trueFrequent.append((key, value))

	return trueFrequent

# Defining the output function.
def writeFile(fileName, inputList):

	## Defining the output array.
    outList = []

    ## Pointer to keep track of the current length.
    currLen = 1

    ## Iterating over the input list.
    for itemSet in inputList:

    	## Process single length items seperately.
        if (len(itemSet) == 1):
            outList.append("('" + itemSet[0] + "')")
            
        elif (len(itemSet) == currLen + 1):
            line = ','.join(outList)
            fileName.write(line + '\n')
            fileName.write('\n')

            currLen += 1
            outList = []
            outList.append(str(itemSet))
        else:
            outList.append(str(itemSet))

    line = ','.join(outList)
    fileName.write(line + '\n')
    fileName.write('\n')

## Defining the number of partitions.
numPartitions = reviewsRDD.getNumPartitions()

## Phase 1 of SON. Obtain the frequent candidates chunk-wise.
phase1Output = reviewsRDD.mapPartitions(lambda basket : aPriori(basket, supportThreshold, numPartitions)).map(lambda f : (tuple(sorted(f)), 1)).reduceByKey(lambda candidate, boolVal : boolVal).keys().collect()
phase1Output = sorted(phase1Output, key = (lambda f: (len(f), f)))

## Phase 2 of SON. Count the frequency of the candidate items and find the true frequents.
phase2Output = reviewsRDD.mapPartitions(lambda basket : generateTrueFrequents(basket, phase1Output)).reduceByKey(lambda candidate, count : candidate + count).filter(lambda f : f[1] >= supportThreshold).keys().collect()
phase2Output = sorted(phase2Output, key = (lambda f: (len(f), f)))

with open(outfilePath, 'w') as outfile:

	outfile.write('Candidates:' + '\n')
	writeFile(outfile, phase1Output)
	
	outfile.write('Frequent Itemsets:' + '\n')
	writeFile(outfile, phase2Output)

tockTime = time.time()

print(tockTime - ticTime)