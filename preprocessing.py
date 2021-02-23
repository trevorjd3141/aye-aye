from os import listdir
from os.path import isfile, join
import pandas as pd
import re

def fetchLines(path):
    file = open(path, 'r')
    return [line.strip().lower() for line in file.readlines() if line.strip() != '']

def cleanText(text):
    # Replace any weird forum chars
    text = re.sub('[<>#|^]', '', text)

    # Remove blank space at the beginning of lines
    text = re.sub('^ *', '', text)

    # Remove extra spaces
    text = re.sub('  +', ' ', text)

    # Attempts to remove footers
    text = re.sub('[ \n]\W{4,}', '', text)

    return text

def allDirectories(path):
    return [file for file in listdir(path) if not isfile(join(path, file))]

def allFiles(path):
    return [file for file in listdir(path) if isfile(join(path, file))]

def extract(path):
    categoryDFs = []
    for directory in allDirectories(path):
        data = []
        for file in allFiles(join(path, directory)):
            filePath = join(path, directory, file)

            # Remove the first two lines as they are irrelevant headers
            text = ' '.join(fetchLines(filePath)[2:])
            text = cleanText(text)
            data.append([directory, file, text])
        categoryDFs.append(pd.DataFrame(data, columns=['Category', 'File', 'Text']))
    concatDF = pd.concat(categoryDFs)
    concatDF.to_csv('data/compiledText.csv', index=False)