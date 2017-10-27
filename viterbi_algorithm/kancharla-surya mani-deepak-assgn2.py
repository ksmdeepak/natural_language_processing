import numpy as np
from collections import Counter, defaultdict,namedtuple
import math
import random
from random import shuffle

random.seed(20170830)
SplitIndices = namedtuple("SplitIndices", ["train", "test"])


def loadData(file):
    with open(file, "r") as file:
        data = []
        sentence=[]
        rowData = []
        for row_index, row in enumerate(file):
            rowData=[]
            for col_index, value in enumerate(row.split("\t")):
                # convert value from string to int
                # strip removes spaces newlines and other pesky characters
                b = value.strip()
                rowData.append(b)
            if(len(rowData)==1):
                data.append(sentence)
                sentence=[]
            else:
                sentence.append(rowData)
        return data
def compute_tagModel_observationTable(trainData):
    tags={}
    words={}
    transition_table=defaultdict(dict)
    observation_table = defaultdict(dict)
    tag_previous='START'
    tag_current= ""
    word_current = ""
    for sentence in trainData:
        for tuple in sentence:
            if tuple[2] in tags:
                tags[tuple[2]] +=1
            else:
                tags[tuple[2]]=1
            if tuple[1] in words:
                words[tuple[1]] +=1
            else:
                words[tuple[1]]=1
            tag_current = tuple[2]
            if tag_previous in transition_table:
                if tag_current in transition_table[tag_previous]:
                    transition_table[tag_previous][tag_current] +=1
                else:
                    transition_table[tag_previous][tag_current]=1
            else:
                transition_table[tag_previous][tag_current]=1
            if tag_current!='.':
                if tag_current in observation_table:
                    if tuple[1] in observation_table[tag_current]:
                        observation_table[tag_current][tuple[1]] += 1
                    else:
                        observation_table[tag_current][tuple[1]] = 1
                else:
                    observation_table[tag_current][tuple[1]] = 1
            tag_previous = tag_current
        tag_previous = 'START'


    #calculating probabilities in transition table and smoothing

    for tag1 in tags:
        smoothing_denominator = tags[tag1] + (len(tags)*len(tags))
        for tag2 in tags:
            if tag1 in transition_table:
                if tag2 in transition_table[tag1]:
                    transition_table[tag1][tag2] = math.log(
                        (transition_table[tag1][tag2] + 1) / smoothing_denominator)
                else:
                    transition_table[tag1][tag2] = math.log((0 + 1) / smoothing_denominator)
            else:
                transition_table[tag1][tag2] = math.log((0 + 1) / smoothing_denominator)
    for tag2 in tags:
        if tag2 in transition_table["START"]:
            transition_table["START"][tag2] = math.log(
                (transition_table["START"][tag2] + 1) / (tags["."] + (len(tags)*len(tags)) ))
        else:
            transition_table["START"][tag2] = math.log((0 + 1) / (tags["."] + (len(tags)*len(tags))))

    #No smmothing done
    # for tag1 in tags:
    #     for tag2 in tags:
    #         if tag1 in transition_table:
    #             if tag2 in transition_table[tag1]:
    #                 transition_table[tag1][tag2] = math.log((transition_table[tag1][tag2]))
    #             else:
    #                 transition_table[tag1][tag2]= -9999
    #         else:
    #             transition_table[tag1][tag2]= -9999
    # for tag2 in tags:
    #     if tag2 in transition_table["START"]:
    #         transition_table["START"][tag2] = math.log((transition_table["START"][tag2]))
    #     else:
    #         transition_table["START"][tag2] = -9999


        # introducing unk tag
    unk_words = []
    count_x=0
    for word in words:
        if words[word] == 1:
            unk_words.append(word)
            count_x += 1
    words["UNK"] = count_x

    for tag in tags:
        observation_table[tag]["UNK"] = 0

    for tag in tags:
        for word in unk_words:
            if tag in observation_table:
                if word in observation_table[tag]:
                    observation_table[tag]["UNK"] += 1
                    observation_table[tag].pop(word)

    for word in unk_words:
        words.pop(word)

    baseline_observation_table = defaultdict(dict)
    for key in observation_table:
        for value in observation_table[key]:
            baseline_observation_table[key][value] = observation_table[key][value]
    #calculating probabilities in observation table

    for tag in tags:
        for word in words:
            if tag in observation_table:
                if word in observation_table[tag] and observation_table[tag][word]!=0:
                    observation_table[tag][word] = math.log(observation_table[tag][word]/tags[tag])
                else:
                    observation_table[tag][word] = -9999
            else:
                observation_table[tag][word] = -9999
    for tag in tags:
        for word in words:
            if tag in baseline_observation_table:
                if word not in baseline_observation_table[tag]:
                    baseline_observation_table[tag][word] = 0
            else:
                baseline_observation_table[tag][word] = 0

    if "." in observation_table:
        observation_table.pop(".")
    if "." in baseline_observation_table:
        baseline_observation_table.pop(".")
    return(transition_table,observation_table,baseline_observation_table,tags,words)


def split_cv(length, num_folds):
    """
    This function splits index [0, length - 1) into num_folds (train, test) tuples.
    """
    splits = [SplitIndices([], []) for _ in range(num_folds)]
    indices = list(range(length))
    random.shuffle(indices)
    fold_length = length / num_folds
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


def cv_performance(data, num_folds,type):
    """This function evaluates average accuracy in cross validation."""
    length = len(data)
    splits = split_cv(length, num_folds)
    accuracy_array = []

    for split in splits:
        accuracy = 0
        train_x = []
        test_x = []
        for x in split.train:
            train_x.append(data[x])
        for x in split.test:
            test_x.append(data[x])
        transition_table, observation_table,baseline_observation_table, tags, words=compute_tagModel_observationTable(train_x)
        hmm = Hmm(train_x,transition_table,observation_table,baseline_observation_table,tags,words)
        if type=="baseline":
            confusion=hmm.confusion_matrix_baseline(test_x)
        elif type=="viterbi":
            confusion=hmm.confusion_matrix(test_x)
        accuracy = hmm.accuracy(confusion)
        accuracy_array.append(accuracy)

    return np.mean(accuracy_array)


class Hmm:
    def __init__(self,trainData,transition_table,observation_table,baseline_observation_table,tags,words):
        self.trainData = trainData
        self.transition_table = transition_table
        self.observation_table = observation_table
        self.baseline_observation_table = baseline_observation_table
        self.tags= tags
        self.words = words

    def viterbi_decoder(self,sequence):
        viterbi = defaultdict(dict)
        backTrack = defaultdict(dict)
        for key in self.observation_table:
            for word in sequence:
                viterbi[key][word] = 0
                backTrack[key][word] = ""

        counter=1
        for word in sequence:
            if(counter==1):
                for key in viterbi:
                    viterbi[key][word] = self.observation_table[key][word] + self.transition_table['START'][key]
                    backTrack[key][word]=0
            else:
                for key in viterbi:
                    max = -99999
                    for key2 in viterbi:
                        value = viterbi[key2][sequence[counter-2]] + self.transition_table[key2][key]
                        if(value>max):
                            max = value
                            backTrack[key][word] = key2
                            # backTrack[key][word].word = sequence[counter-1]
                    viterbi[key][word] = max + self.observation_table[key][word]
            counter += 1

        max = -99999
        maxKey=''
        for key in viterbi:
            value = viterbi[key][sequence[len(sequence)-1]] + self.transition_table[key]['.']
            if(value>max):
                max=value
                maxKey = key
        tagSequence = []
        tagSequence.append(maxKey)
        traverseKey = maxKey

        for (counter, word) in enumerate(reversed(sequence)):
            traverseKey = backTrack[traverseKey][word]
            if counter==len(sequence)-1:
                break
            tagSequence.append(traverseKey)
        return tagSequence

    def baseline_decoder(self,sequence):
        output=[]
        for word in sequence:
            max=-9999
            maxKey=''
            for tag in self.baseline_observation_table:
                if(self.baseline_observation_table[tag][word]>max):
                    max=self.baseline_observation_table[tag][word]
                    maxKey=tag
            output.append(maxKey)
        return output


    def confusion_matrix(self, testData):
        d = defaultdict(dict)
        for x in self.tags:
            for y in self.tags:
                d[x][y]=0
        counter=0
        cc=0
        for sentence in testData:
            test_sentence=[]
            test_tags=[]
            for tuple in sentence:
                if tuple[2]!=".":
                    if tuple[1] in  self.words:
                        test_sentence.append(tuple[1])
                    else:
                        counter+=1
                        test_sentence.append("UNK")
                    test_tags.append(tuple[2])
            output = self.viterbi_decoder(test_sentence)
            count= len(output)-1
            for x in test_tags:
                d[x][output[count]] +=1
                count -=1
        return d

    def confusion_matrix_baseline(self, testData):
        d = defaultdict(dict)
        for x in self.tags:
            for y in self.tags:
                d[x][y]=0
        counter=0
        for sentence in testData:
            test_sentence=[]
            test_tags=[]
            for tuple in sentence:
                if tuple[2]!=".":
                    if tuple[1] in self.words:
                        test_sentence.append(tuple[1])
                    else:
                        counter+=1
                        test_sentence.append("UNK")
                    test_tags.append(tuple[2])
            output = self.baseline_decoder(test_sentence)
            count= 0
            for x in test_tags:
                if output[count] in d[x]:
                    d[x][output[count]] +=1
                count +=1
        return d

    @staticmethod
    def accuracy(confusion_matrix):
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
        f=open("kancharla-surya mani deepak-assgn2-test-output.txt","w")
        test_sentence=[]
        for sentence in testData:
            test_sentence=[]
            for x in sentence:
                if x[1]!=".":
                    if x[1] in  self.words:
                        test_sentence.append(x[1])
                    else:
                        test_sentence.append("UNK")
            output = self.viterbi_decoder(test_sentence)
            output.insert(0,".")
            count= len(output)-1
            for tuple in sentence:
                f.write(tuple[0]+"\t"+tuple[1]+"\t"+output[count]+"\n")
                count-=1
            f.write("\n")
if __name__ == "__main__":
    data = loadData("berp-POS-training.txt")
    shuffle(data)
    transition_table,observation_table,baseline_observation_table,tags,words=compute_tagModel_observationTable(data)
    hmm = Hmm(data,transition_table,observation_table,baseline_observation_table,tags,words)

    # code for generating output file for given test data
    # testData = loadData("assgn2-test-set.txt")
    # hmm.write_output(testData)

    #k-fold testing
    k=5
    accuracy = cv_performance(data,k,"baseline")
    print("%d fold accuracy for baseline: " %k )
    print(accuracy)
    accuracy = cv_performance(data, k, "viterbi")
    print("%d fold accuracy for viterbi: " % k)
    print(accuracy)