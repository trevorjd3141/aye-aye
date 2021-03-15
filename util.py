from nltk.corpus import wordnet
from os import listdir
from os.path import isfile, join
from collections import defaultdict

def fetch_lines(path):
    file = open(path, 'r')
    return [line.strip().lower() for line in file.readlines() if line.strip() != '']

def categorize(word):
    choices = defaultdict(int)
    matches = wordnet.synsets(word)
    lexnames = [match.lexname() for match in matches]
    if not matches:
        return 'None'
    elif 'noun.food' in lexnames:
        return 'noun.food'
    else:
        return lexnames[0]