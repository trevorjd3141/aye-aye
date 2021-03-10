from os.path import join

import grader
import preprocessing
import recall
import basilisk
import seedSelector

OUTPUT_PATH = 'output\\'
DATA_PATH = 'data\\'
PROCESSED_DATA_NAME = 'processed_data.csv'
PROCESSED_DATA = join(DATA_PATH, PROCESSED_DATA_NAME)
PICKLE = join(DATA_PATH, 'extractions.p')
DOCS = join(DATA_PATH, 'docs.p')
CATEGORY = 'noun.food'
OPTIONS = {
    'preprocessing': False,
    'seed-selection': False,
    'recall': False,
    'extraction': True,
    'grading': True
}

def main():
    if OPTIONS['preprocessing']:
        preprocessing.compile(DATA_PATH, PROCESSED_DATA_PATH)

    if OPTIONS['seed-selection']:
        seedSelector.compileCommonWords(PROCESSED_DATA, CATEGORY)

    if OPTIONS['recall']:
        recall.compileCategoryWords(PROCESSED_DATA, CATEGORY)

    if OPTIONS['extraction']:
        basilisk.basilisk(CATEGORY, OUTPUT_PATH, PROCESSED_DATA, PICKLE, DOCS, True)

    if OPTIONS['grading']:
        grader.read(OUTPUT_PATH, CATEGORY)

if __name__ == "__main__":
    main()