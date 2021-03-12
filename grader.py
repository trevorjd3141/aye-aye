import util
import pandas as pd

def calculateFScore(recall, precision):
    if (recall + precision) == 0:
        return 0.00
    score = (2 * recall * precision)/(precision + recall)
    return round(score, 3)

# Reads in a results file and tests its accuracy
def read(path, category):
    words = util.fetchLines(f'{path}{category}.txt')

    data = []
    total = len(words)
    correct = 0
    for word in words:
        data.append([word, util.categorize(word), category, util.categorize(word) == category])
        if util.categorize(word) == category:
            correct += 1
    precision = correct/total
    accuracy = 100 * precision
    df = pd.DataFrame(data, columns=['Word', 'Correct Category', 'Guessed Category', 'Correct Guess'])
    df.to_csv(f'{path}{category}-results.csv', index=False)

    print(f'Correctly guessed {correct} out of {total} words')
    print(f'For a total accuracy of {round(accuracy, 3)}%')
    print(f'Labeled matches sent to {category}-results.csv')
    print()

    allCategoryWords = util.fetchLines(f'recall\\{category}.txt')
    totalCategoryWords = len(allCategoryWords)
    recall = correct/totalCategoryWords
    fScore = calculateFScore(recall, precision)

    print(f'Final Precision: {round(precision, 3)}')
    print(f'Final Recall: {round(recall, 3)}')
    print(f'Final F-Score: {round(fScore, 3)}')