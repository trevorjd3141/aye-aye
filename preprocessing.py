from os.path import join
import pandas as pd
import re
import util
import spacy
nlp = spacy.load("en_core_web_md")

SAMPLE_SIZE = 4000
ANNOTATIONS_PATH = 'data\\annotations\\'

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

def extract(path):
    categoryDFs = []
    allText = ''
    for directory in util.allDirectories(path):
        data = []
        for file in util.allFiles(join(path, directory)):
            filePath = join(path, directory, file)

            # Remove the first two lines as they are irrelevant headers
            text = ' '.join(util.fetchLines(filePath)[2:])
            text = cleanText(text)
            data.append([directory, file, text])
            allText += f'\n{text}'
        categoryDFs.append(pd.DataFrame(data, columns=['Category', 'File', 'Text']))
        print(directory)
    concatDF = pd.concat(categoryDFs)
    concatDF = concatDF.sample(n=SAMPLE_SIZE)
    concatDF.to_csv('data/compiledText.csv', index=False)

    for text, file in zip(concatDF['Text'], concatDF['File']):
        doc = nlp(text)
        doc.to_disk(f'{ANNOTATIONS_PATH}{file}')
    
    textOutput = open(f'data/compiledText.txt', 'w')
    textOutput.write(allText) 
    textOutput.close() 