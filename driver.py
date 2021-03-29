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
PICKLE_NAME = 'extractions.p'
PROCESSED_DATA = join(DATA_PATH, PROCESSED_DATA_NAME)
TRUNCATED_DATA = join(DATA_PATH, TRUNCATED_DATA_NAME)
PICKLE_PATH = join(DATA_PATH, PICKLE_NAME)
DOCS = join(DATA_PATH, 'docs.p')
SETTINGS = [
    {'NAME': 'noun.person', 'LEFT_TOKENS': 2, 'PARENT_TOKENS': 3,
    'RIGHT_TOKENS': 1, 'LEFT_SIBLING': True, 'RIGHT_SIBLING': True,
    'MIN_PATTERN_COMPLEXITY': 2, 'MAX_PATTERN_COMPLEXITY': 3},
    {'NAME': 'noun.food', 'LEFT_TOKENS': 2, 'PARENT_TOKENS': 2,
    'RIGHT_TOKENS': 1, 'LEFT_SIBLING': True, 'RIGHT_SIBLING': True,
    'MIN_PATTERN_COMPLEXITY': 2, 'MAX_PATTERN_COMPLEXITY': 3},
]
TRUNCATED_COUNT = 30000
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
        truncation.truncate(PROCESSED_DATA, TRUNCATED_DATA, TRUNCATED_COUNT)

    if OPTIONS['seed-selection']:
        for setting in SETTINGS:
            category = setting['NAME']
            seed_selector.compile_common_words(TRUNCATED_DATA, category)
    
    if OPTIONS['recall']:
        for setting in SETTINGS:
            category = setting['NAME']
            recall.compile_setting_words(TRUNCATED_DATA, category)

    if OPTIONS['extraction']:
        for setting in SETTINGS:
            category = setting['NAME']
            aye_aye.aye_aye(setting, OUTPUT_PATH, TRUNCATED_DATA, PICKLE_PATH, DOCS, True)

    if OPTIONS['grading']:
        for setting in SETTINGS:
            category = setting['NAME']
            grader.read(OUTPUT_PATH, category)

if __name__ == "__main__":
    main()