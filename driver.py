from os.path import join

import preprocessing
import truncation
import recall
import seed_selector
import aye_aye
import grader

# Data Paths
OUTPUT_PATH = 'output\\'
RAW_DATA_PATH = 'data\\raw\\'
PICKLE_PATH = 'data\\memoization\\'
INTERMEDIATE_DATA_PATH = 'data\\intermediate\\'
PROCESSED_DATA_NAME = 'processed_data.csv'
PROCESSED_DATA = join(RAW_DATA_PATH, PROCESSED_DATA_NAME)
TRUNCATED_DATA = join(RAW_DATA_PATH, 'truncated_data.csv')
EXTRACTIONS_PATH = join(PICKLE_PATH, 'extractions.p')
DOCS_PATH = join(PICKLE_PATH, 'docs.p')

# Truncation Settings
TRUNCATED_COUNT = 30000

# Extraction Settings
MULTI_CATEGORICAL_LEARNING = True
SETTINGS = [
    {'NAME': 'noun.person', 'LEFT_TOKENS': 2, 'PARENT_TOKENS': 3,
    'RIGHT_TOKENS': 2, 'LEFT_SIBLING': True, 'RIGHT_SIBLING': True,
    'MIN_PATTERN_COMPLEXITY': 1, 'MAX_PATTERN_COMPLEXITY': 4},
    {'NAME': 'noun.food', 'LEFT_TOKENS': 2, 'PARENT_TOKENS': 3,
    'RIGHT_TOKENS': 2, 'LEFT_SIBLING': True, 'RIGHT_SIBLING': True,
    'MIN_PATTERN_COMPLEXITY': 1, 'MAX_PATTERN_COMPLEXITY': 4},
]

# Driver Settings
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
        if MULTI_CATEGORICAL_LEARNING:
            aye_aye.aye_aye(SETTINGS, INTERMEDIATE_DATA_PATH, TRUNCATED_DATA, EXTRACTIONS_PATH, DOCS_PATH, True)
        else:
            for setting in SETTINGS:
                aye_aye.aye_aye([setting], INTERMEDIATE_DATA_PATH, TRUNCATED_DATA, EXTRACTIONS_PATH, DOCS_PATH, True)

    if OPTIONS['grading']:
        for setting in SETTINGS:
            category = setting['NAME']
            grader.read(INTERMEDIATE_DATA_PATH, OUTPUT_PATH, category)

if __name__ == "__main__":
    main()