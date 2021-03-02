import grader
import preprocessing
import recall
import basilisk

OUTPUT_PATH = 'output\\'
NEWS_PATH = 'data\\20news\\'
"""
CATEGORIES = ['noun.person', 'noun.animal', 'noun.plant',
                'noun.location', 'noun.group', 'noun.food']
"""
CATEGORIES = ['noun.person']
OPTIONS = {
    'recall': False,
    'preprocessing': False,
    'extraction': True,
    'grading': True
}

def main():
    if OPTIONS['preprocessing']:
        preprocessing.extract(NEWS_PATH)

    if OPTIONS['recall']:
        for category in CATEGORIES:
            recall.compileCategoryWords(NEWS_PATH, category)

    if OPTIONS['extraction']:
        for category in CATEGORIES:
            basilisk.basilisk(category, OUTPUT_PATH, True)

    if OPTIONS['grading']:
        for category in CATEGORIES:
            grader.read(OUTPUT_PATH, category)

if __name__ == "__main__":
    main()