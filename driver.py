from os.path import join

import grader
import preprocessing
import recall
import basilisk

OUTPUT_PATH = 'output\\'
PROCESSED_DATA_NAME = 'processed_data.csv'
DATA_PATH = 'data\\'
CATEGORY = 'noun.food'
OPTIONS = {
    'preprocessing': True,
    'recall': True,
    'extraction': False,
    'grading': False
}

def main():
    if OPTIONS['preprocessing']:
        preprocessing.compile(DATA_PATH, PROCESSED_DATA_NAME)

    if OPTIONS['recall']:
        recall.compileCategoryWords(join(DATA_PATH, PROCESSED_DATA_NAME), CATEGORY)

    if OPTIONS['extraction']:
        basilisk.basilisk(CATEGORY, OUTPUT_PATH, True)

    if OPTIONS['grading']:
        grader.read(OUTPUT_PATH, CATEGORY)

if __name__ == "__main__":
    main()