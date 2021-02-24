import grader
import preprocessing
import recall

OUTPUT_PATH = 'output\\test.txt'
NEWS_PATH = 'data\\20news\\'
CATEGORIES = ['noun.person', 'noun.animal', 'noun.plant',
                'noun.location', 'noun.group', 'noun.food']
OPTIONS = {
    'recall': False,
    'preprocessing': False,
    'grading': True
}

def main():
    if OPTIONS['preprocessing']:
        preprocessing.extract(NEWS_PATH)

    if OPTIONS['recall']:
        for category in CATEGORIES:
            recall.compileCategoryWords(NEWS_PATH, category)

    if OPTIONS['grading']:
        grader.read(OUTPUT_PATH, 'noun.person')

if __name__ == "__main__":
    main()