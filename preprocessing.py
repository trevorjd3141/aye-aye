from os.path import join
import pandas as pd
import re
import util

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
            allText += f'\n {text}'
        categoryDFs.append(pd.DataFrame(data, columns=['Category', 'File', 'Text']))
    concatDF = pd.concat(categoryDFs)
    concatDF.to_csv('data/compiledText.csv', index=False)
    
    textOutput = open(f'data/compiledText.txt', 'w')
    textOutput.write(allText) 
    textOutput.close() 