from os.path import join

import preprocessing
import truncation
import recall
import seedSelector
import basilisk
import grader

OUTPUT_PATH = 'output\\'
DATA_PATH = 'data\\'
PROCESSED_DATA_NAME = 'processed_data.csv'
TRUNCATED_DATA_NAME = 'truncated_data.csv'
PROCESSED_DATA = join(DATA_PATH, PROCESSED_DATA_NAME)
TRUNCATED_DATA = join(DATA_PATH, TRUNCATED_DATA_NAME)
PICKLE = join(DATA_PATH, 'extractions.p')
DOCS = join(DATA_PATH, 'docs.p')
CATEGORY = 'noun.food'
OPTIONS = {
    'preprocessing': False,
    'truncation': False,
    'seed-selection': False,
    'recall': False,
    'extraction': True,
    'grading': True
}

def main():
    if OPTIONS['preprocessing']:
        preprocessing.compile(DATA_PATH, PROCESSED_DATA_NAME)

    if OPTIONS['truncation']:
        truncation.truncate(PROCESSED_DATA, TRUNCATED_DATA)

    if OPTIONS['seed-selection']:
        seedSelector.compileCommonWords(TRUNCATED_DATA, CATEGORY)

    if OPTIONS['recall']:
        recall.compileCategoryWords(TRUNCATED_DATA, CATEGORY)

    if OPTIONS['extraction']:
        basilisk.basilisk(CATEGORY, OUTPUT_PATH, TRUNCATED_DATA, PICKLE, DOCS, True)

    if OPTIONS['grading']:
        grader.read(OUTPUT_PATH, CATEGORY)

if __name__ == "__main__":
    main()