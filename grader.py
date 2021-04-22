import util
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
import pickle

SKIP = 5

def draw_graph(category, data, path):
    total_words = len(data)
    important_markers = list(range(SKIP, total_words + SKIP, SKIP))
    points = []
    for marker in important_markers:
        if marker <= total_words:
            precision = round(len([result for result in data[:marker] if result[-1]])/marker, 2)
            points.append((marker, precision))

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Total Words', ylabel='Precision', title=f'{category.upper()} Precision Graph')
    ax.grid()
    plt.xlim([important_markers[0], important_markers[-1]])

    fig.savefig(path)

def units_to_string(units):
    return ', '.join([str(unit) for unit in units]) if len(units) > 0 else ''

def unit_to_string(unit):
    return '' if unit == () else str(unit)

def analyze_patterns(patterns, path):
    data = []
    for pattern in patterns:
        target_dep, lefts, parents, rights, left_sibling, right_sibling = pattern
        datum = [target_dep, units_to_string(lefts), units_to_string(parents), units_to_string(rights), unit_to_string(left_sibling), unit_to_string(right_sibling)]
        data.append(datum)
    df = pd.DataFrame(data, columns=['Target Depedency', 'Left Units', 'Ancestors', 'Right Units', 'Left Sibling', 'Right Sibling'])
    df.to_csv(path, index=False)

def calculate_f_score(recall, precision):
    if (recall + precision) == 0:
        return 0.000
    score = (2 * recall * precision)/(precision + recall)
    return round(score, 3)

# Driver for the grader
def read(input_path, output_path, category):
    words = util.fetch_lines(f'{input_path}/{category}.txt')

    data = []
    total = len(words)
    correct = 0
    for word in words:
        data.append([word, util.categorize(word, category), category, util.categorize(word, category) == category])
        if util.categorize(word, category) == category:
            correct += 1
    overall_precision = correct/total
    accuracy = 100 * overall_precision
    df = pd.DataFrame(data, columns=['Word', 'Correct Category', 'Guessed Category', 'Correct Guess'])
    df.to_csv(f'{output_path}/{category}-results.csv', index=False)

    print(f'Correctly guessed {correct} out of {total} words')
    print(f'For a total accuracy of {round(accuracy, 3)}%')
    print(f'Labeled matches sent to {category}-results.csv')
    print()

    draw_graph(category, data, f'{output_path}/{category}-precision.png')
    with open(f'{input_path}/{category}-patterns.p', 'rb') as file: 
        patterns = pickle.load(file)
    analyze_patterns(patterns, f'{output_path}/{category}-patterns.csv')
    print()

    all_category_words = util.fetch_lines(f'recall/{category}.txt')
    total_category_words = len(all_category_words)
    recall = correct/total_category_words
    f_score = calculate_f_score(recall, overall_precision)

    print(f'Final Overall Precision: {round(overall_precision, 3)}')
    print(f'Final Overall Recall: {round(recall, 3)}')
    print(f'Final Overall F-Score: {round(f_score, 3)}')
    print()