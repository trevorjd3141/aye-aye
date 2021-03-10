import util
from os.path import join
import pandas as pd

# Need to normalize the words before adding them to the set

def compileCategoryWords(path, category):
    categoryWords = set()
    texts = pd.read_csv(path, squeeze=True)
    for text in texts:
        words = text.split()
        for word in words:
            if util.categorize(word) == category:
                categoryWords.add(word.lower())
    output = open(f'recall/{category}.txt', 'w')
    for word in categoryWords:
        output.write(f'{word}\n') 
    output.close() 