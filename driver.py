from os.path import join

import preprocessing
import truncation
import recall
import seed_selector
import aye_aye
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
TRUNCATED_COUNT = 16000
OPTIONS = {
    'preprocessing': False,
    'truncation': False,
    'seed-selection': False,
    'recall': False,
    'extraction': True,
    'grading': False
}

def main():
    if OPTIONS['preprocessing']:
        preprocessing.compile(DATA_PATH, PROCESSED_DATA_NAME)

    if OPTIONS['truncation']:
        truncation.truncate(PROCESSED_DATA, TRUNCATED_DATA, TRUNCATED_COUNT)

    if OPTIONS['seed-selection']:
        seed_selector.compile_common_words(TRUNCATED_DATA, CATEGORY)

    if OPTIONS['recall']:
        recall.compile_category_words(TRUNCATED_DATA, CATEGORY)

    if OPTIONS['extraction']:
        aye_aye.aye_aye(CATEGORY, OUTPUT_PATH, TRUNCATED_DATA, PICKLE, DOCS, True)

    if OPTIONS['grading']:
        grader.read(OUTPUT_PATH, CATEGORY)

if __name__ == "__main__":
    main()