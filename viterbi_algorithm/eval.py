import sys

def eval(keys, predictions):
    """ Simple minded eval system: file1 is gold standard, file2 is system outputs. Result is straight accuracy. """

    count = 0.0
    correct = 0.0

    for key, prediction in zip(keys, predictions):
        key = key.strip()
        prediction = prediction.strip()
        if key == '': continue
        count += 1
        if key == prediction: correct += 1

    print("Evaluated ", count, " tags.")
    print("Accuracy is: ", correct / count)

if __name__ == "__main__":
     keys = open("output.txt")
     predictions = open("kancharla-surya mani deepak-assgn2-test-output.txt")
     eval(keys, predictions)
