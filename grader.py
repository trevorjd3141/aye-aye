import util
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

IMPORTANT_MARKERS = (20, 40, 60, 80, 100)

def draw_graph(points, path):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Total Words', ylabel='Precision', title='Precision for Top-K Words')
    ax.grid()

    fig.savefig(path)

def calculate_f_score(recall, precision):
    if (recall + precision) == 0:
        return 0.00
    score = (2 * recall * precision)/(precision + recall)
    return round(score, 3)

# Reads in a results file and tests its accuracy
def read(path, category):
    words = util.fetch_lines(f'{path}{category}.txt')

    data = []
    total = len(words)
    correct = 0
    for word in words:
        data.append([word, util.categorize(word), category, util.categorize(word) == category])
        if util.categorize(word) == category:
            correct += 1
    overall_precision = correct/total
    accuracy = 100 * overall_precision
    df = pd.DataFrame(data, columns=['Word', 'Correct Category', 'Guessed Category', 'Correct Guess'])
    df.to_csv(f'{path}{category}-results.csv', index=False)

    print(f'Correctly guessed {correct} out of {total} words')
    print(f'For a total accuracy of {round(accuracy, 3)}%')
    print(f'Labeled matches sent to {category}-results.csv')
    print()

    points = []
    for marker in IMPORTANT_MARKERS:
        if marker < len(data):
            precision = round(len([result for result in data[:marker] if result[-1]])/total,2)
            points.append((marker, precision))
            print(f'Precision for the top {marker} extracted words is {precision}')
    draw_graph(points, f'{path}{category}-precision.png')
    print()

    all_category_words = util.fetch_lines(f'recall\\{category}.txt')
    total_category_words = len(all_category_words)
    recall = correct/total_category_words
    f_score = calculate_f_score(recall, overall_precision)

    print(f'Final Overall Precision: {round(overall_precision, 3)}')
    print(f'Final Overall Recall: {round(recall, 3)}')
    print(f'Final Overall F-Score: {round(f_score, 3)}')