import numpy as np
from collections import Counter, defaultdict,namedtuple
import math
import random
from random import shuffle
import re

random.seed(20170830)
SplitIndices = namedtuple("SplitIndices", ["train", "test"])

def loadData(file):
    with open(file, "r",encoding="utf8") as file:
        data = []
        rowData = []
        for row in file:
            rowData=[]
            for value in row.split("\t"):
                # convert value from string to int
                # strip removes spaces newlines and other pesky characters
                b = value.strip()
                rowData.append(b)
            data.append(rowData)
        return data

def split_cv(length, num_folds):
    """
    This function splits index [0, length - 1) into num_folds (train, test) tuples.
    """
    splits = [SplitIndices([], []) for _ in range(num_folds)]
    indices = list(range(length))
    random.shuffle(indices)
    fold_length = (int)(length / num_folds)

    for y in range(1, num_folds + 1):
        fold_n = 1
        counter = 0
        for x in indices:
            if fold_n == y:
                splits[y - 1].test.append(x)
            else:
                splits[y - 1].train.append(x)
            counter = counter + 1
            if counter % fold_length == 0:
                fold_n = fold_n + 1
    return splits


def cv_performance(posTrainData,negTrainData, num_folds):
    """This function evaluates average accuracy in cross validation."""
    length = len(negTrainData)
    splits = split_cv(length, num_folds)
    accuracy_array = []
    for split in splits:
        accuracy = 0
        train_pos = []
        train_neg = []
        test_neg = []
        test_pos = []
        for x in split.train:
            train_pos.append(posTrainData[x])
            train_neg.append(negTrainData[x])
        for x in split.test:
            test_pos.append(posTrainData[x])
            test_neg.append(negTrainData[x])
        nb = Nb(train_pos,train_neg)
        confusion=nb.confusion_matrix(test_pos,test_neg)
        accuracy = nb.accuracy(confusion)
        accuracy_array.append(accuracy)

    return accuracy_array

class Nb:
    def __init__(self,posTrainData,negTrainData):
        self.posTrainData = posTrainData
        self.negTrainData=negTrainData
        self.observation_table = defaultdict(dict)
        self.observation_table["pos"]={}
        self.observation_table["neg"]={}
        self.tags= {}
        self.tags["pos"] = len(posTrainData)
        self.tags["neg"] = len(negTrainData)
        self.words = {}
        # self.pattern = "\w\S*"
        # self.pattern = "([a-zA-Z]+|[0-9]+)\S*"
        self.countWords={}
        self.countWords["pos"] = 0
        self.countWords["neg"] = 0
        self.trainModel()

    def trainModel(self):
        for data in self.posTrainData:
            words = self.extractWords(data[1])
            # words = re.findall(self.pattern,data[1])
            for word in words:
                self.countWords["pos"]+=1
                if word in self.words:
                    self.words[word]+=1
                else:
                    self.words[word] = 1
                if word in self.observation_table["pos"]:
                    self.observation_table["pos"][word]+=1
                else:
                    self.observation_table["pos"][word]=1
        for data in self.negTrainData:
            words = self.extractWords(data[1])
            for word in words:
                self.countWords["neg"]+=1
                if word in self.words:
                    self.words[word]+=1
                else:
                    self.words[word] = 1
                if word in self.observation_table["neg"]:
                    self.observation_table["neg"][word]+=1
                else:
                    self.observation_table["neg"][word]=1

        #smoothing and unknowns to be dealed.
        for key in self.observation_table:
            for word in self.words:
                if word in self.observation_table[key]:
                    self.observation_table[key][word]=math.log((self.observation_table[key][word]+1)/(self.countWords[key]+len(self.words)))
                else:
                    self.observation_table[key][word]=math.log(1/(self.countWords[key]+len(self.words)))
    def extractWords(self,words):
        words=words.replace(',','')
        words=words.replace('.','')
        return words.split()
    def computeClass(self,record):
        words = self.extractWords(record)
        negProb = 0
        posProb = 0
        for word in words:
            if word in self.words:
                negProb = negProb + self.observation_table["neg"][word]
                posProb = posProb + self.observation_table["pos"][word]
            # need to be done based on unknown word dealing
            # else:
            #     negProb = negProb + self.observation_table["neg"]["UNK"]
        if(posProb > negProb):
            return 1
        else:
            return 0

    def confusion_matrix(self, testPos,testNeg):
        d = defaultdict(dict)
        for x in [0,1]:
            for y in [0,1]:
                d[x][y] = 0
        for review in testPos:
            d[1][self.computeClass(review[1])]+=1
        for review in testNeg:
            d[0][self.computeClass(review[1])]+=1
        return d

    def accuracy(self,confusion_matrix):
        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii][ii]
        if total > 0:
            return float(correct) / float(total)
        else:
            return 0.0
    def write_output(self,testData):
        f=open("kancharla-surya mani deepak-assgn3-out.txt","w")
        for record in testData:
            output = self.computeClass(record[1])
            if output==0:
                output="NEG"
            else:
                output="POS"
            f.write(record[0]+"\t"+output)
            f.write("\n")

if __name__ == "__main__":
    posTrainData = loadData("hotelPosT-train.txt")
    negTrainData = loadData("hotelNegT-train.txt")
    testData = loadData("HW3-testset.txt")
    # code for generating output file for given test data
    # nb = Nb(posTrainData,negTrainData)
    # nb.write_output(testData)

    # k-fold testing
    k=5
    accuracy = cv_performance(posTrainData,negTrainData, k)
    print(accuracy)
    print(np.mean(accuracy))