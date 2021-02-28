import grader
import preprocessing
import recall
import basilisk

OUTPUT_PATH = 'output\\test.txt'
NEWS_PATH = 'data\\20news\\'
CATEGORIES = ['noun.person', 'noun.animal', 'noun.plant',
                'noun.location', 'noun.group', 'noun.food']
OPTIONS = {
    'recall': False,
    'preprocessing': True,
    'extraction': False,
    'grading': False
}

def main():
    if OPTIONS['preprocessing']:
        preprocessing.extract(NEWS_PATH)

    if OPTIONS['recall']:
        for category in CATEGORIES:
            recall.compileCategoryWords(NEWS_PATH, category)

    if OPTIONS['extraction']:
        for category in CATEGORIES:
            basilisk.basilisk(category)

    if OPTIONS['grading']:
        grader.read(OUTPUT_PATH, 'noun.person')

if __name__ == "__main__":
    main()