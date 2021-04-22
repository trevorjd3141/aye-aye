import util
import pandas as pd

# Need to normalize the words before adding them to the set
def compile_category_words(path, category):
    category_words = set()
    texts = pd.read_csv(path, squeeze=True)
    for text in texts:
        words = text.split()
        for word in words:
            if util.categorize(word, category) == category:
                category_words.add(word.lower())
    output = open(f'recall/{category}.txt', 'w')
    for word in category_words:
        output.write(f'{word}\n') 
    output.close() 