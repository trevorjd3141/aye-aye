from os.path import join

import preprocessing
import truncation
import recall
import seed_selector
import aye_aye
import grader

# Data Paths
OUTPUT_PATH = 'output\\'
DATA_PATH = 'data\\'
RAW_DATA_PATH = join(DATA_PATH, 'raw')
PICKLE_PATH = join(DATA_PATH, 'memoization')
INTERMEDIATE_DATA_PATH = join(DATA_PATH, 'intermediate')
PROCESSED_DATA_NAME = 'processed_data.csv'
PROCESSED_DATA = join(RAW_DATA_PATH, PROCESSED_DATA_NAME)
TRUNCATED_DATA = join(RAW_DATA_PATH, 'truncated_data.csv')
EXTRACTIONS_PATH = join(PICKLE_PATH, 'extractions.p')
DOCS_PATH = join(PICKLE_PATH, 'docs.p')
DRIFT_PATH = join(PICKLE_PATH, 'drifts.p')

# Truncation Settings
TRUNCATED_COUNT = 30000

# Extraction Settings
SEMANTIC_DRIFT_FILTERING = False
MULTI_CATEGORICAL_LEARNING = False
MUTUAL_EXCLUSION = False
SETTINGS = [
    {'NAME': 'noun.person', 'LEFT_TOKENS': 0, 'PARENT_TOKENS': 1,
    'RIGHT_TOKENS': 0, 'LEFT_SIBLING': False, 'RIGHT_SIBLING': False,
    'MIN_PATTERN_COMPLEXITY': 1, 'MAX_PATTERN_COMPLEXITY': 1},
    {'NAME': 'noun.food', 'LEFT_TOKENS': 0, 'PARENT_TOKENS': 1,
    'RIGHT_TOKENS': 0, 'LEFT_SIBLING': False, 'RIGHT_SIBLING': False,
    'MIN_PATTERN_COMPLEXITY': 1, 'MAX_PATTERN_COMPLEXITY': 1},
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
        preprocessing.compile(RAW_DATA_PATH, PROCESSED_DATA_NAME)

    if OPTIONS['truncation']:
        truncation.truncate(PROCESSED_DATA, TRUNCATED_DATA, TRUNCATED_COUNT)

    if OPTIONS['seed-selection']:
        for setting in SETTINGS:
            category = setting['NAME']
            seed_selector.compile_common_words(TRUNCATED_DATA, category)
    
    if OPTIONS['recall']:
        for setting in SETTINGS:
            category = setting['NAME']
            recall.compile_category_words(TRUNCATED_DATA, category)

    if OPTIONS['extraction']:
        if MULTI_CATEGORICAL_LEARNING:
            aye_aye.aye_aye(SETTINGS, INTERMEDIATE_DATA_PATH, TRUNCATED_DATA, EXTRACTIONS_PATH, DOCS_PATH, DRIFT_PATH, MUTUAL_EXCLUSION, SEMANTIC_DRIFT_FILTERING, True)
        else:
            for setting in SETTINGS:
                aye_aye.aye_aye([setting], INTERMEDIATE_DATA_PATH, TRUNCATED_DATA, EXTRACTIONS_PATH, DOCS_PATH, DRIFT_PATH, MUTUAL_EXCLUSION, SEMANTIC_DRIFT_FILTERING, True)

    if OPTIONS['grading']:
        for setting in SETTINGS:
            category = setting['NAME']
            grader.read(INTERMEDIATE_DATA_PATH, OUTPUT_PATH, category)

if __name__ == "__main__":
    main()