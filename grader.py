from nltk.corpus import wordnet
import pandas as pd

def fetchWords(fn):
    file = open(fn, 'r')
    return [line.strip().lower() for line in file.readlines()]

def categorize(word):
    matches = wordnet.synsets(word)
    category = matches[0].lexname() if matches else 'None'
    return category

# Reads in a results file and tests its accuracy
def read(file, category):
    words = fetchWords(file)

    data = []
    total = len(words)
    correct = 0
    for word in words:
        data.append([word, categorize(word), category, categorize(word) == category])
        if categorize(word) == category:
            correct += 1
    accuracy = 100 * correct/total
    df = pd.DataFrame(data, columns=['Word', 'Guessed Category', 'Correct Category', 'Correct Guess'])
    df.to_csv(f'output/{category}-results.csv', index=False)

    print(f'Correctly guessed {correct} out of {total} words')
    print(f'For a total accuracy of {round(accuracy, 2)}%')
    print(f'Labeled matches sent to {category}-results.csv')