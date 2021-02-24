from nltk.corpus import wordnet
from os import listdir
from os.path import isfile, join
from collections import defaultdict

def fetchLines(path):
    file = open(path, 'r')
    return [line.strip().lower() for line in file.readlines() if line.strip() != '']

def categorize(word):
    choices = defaultdict(int)
    matches = wordnet.synsets(word)
    return matches[0].lexname() if matches else 'None'

def allDirectories(path):
    return [file for file in listdir(path) if not isfile(join(path, file))]

def allFiles(path):
    return [file for file in listdir(path) if isfile(join(path, file))]