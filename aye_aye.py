import pandas as pd
import math
import spacy
from spacy.matcher import DependencyMatcher
nlp = spacy.load("en_core_web_sm")
matcher = DependencyMatcher(nlp.vocab)
from datetime import datetime
from collections import defaultdict
import pickle
from os.path import isfile
import itertools
from operator import itemgetter
import numpy as np

import util
import pattern_generation

PATTERN_POOL_INIT_SIZE = 20
PATTERN_POOL_SIZE_INCREASE = 2
WORDS_PER_ROUND = 10
MAX_PATTERN_USES = 2
LOOPS = 10
MAX_TEXT_SIZE = 500
MIN_EXTRACTIONS_PER_PATTERN = 3

# Options for filtering what patterns the generator will return
LEFT_TOKENS=2
PARENT_TOKENS=2
RIGHT_TOKENS=1
MIN_PATTERN_COMPLEXITY=2
MAX_PATTERN_COMPLEXITY=3

def avg_log(patterns, category, lexicon):
    # Creates a list of lists containing only category members
    avg_log_members_extracted = 0
    for pattern in patterns:
        category_words = [word for word in pattern[1] if word in lexicon]
        avg_log_members_extracted += math.log(len(category_words)+1, 2)
    p = len(patterns)

    score = avg_log_members_extracted/p
    return round(score, 3)

def r_log_f(words, category, lexicon):
    n = len(words)
    f = len([word for word in words if word in lexicon])

    if n < MIN_EXTRACTIONS_PER_PATTERN or f == 0:
        score = 0
    else:
        score = (f/n)*math.log(f,2)

    return round(score, 3)

def convert_to_dependency_pattern(original_pattern):
    target, lefts, parents, rights = original_pattern
    target_text, target_dep = target
    target_id = f'{target_text}-{target_dep}'
    dependency_pattern = [{'RIGHT_ID': target_id, 'RIGHT_ATTRS': {'DEP': target_dep}}]

    increment = 0
    for unit in itertools.chain(lefts, rights):
        unit_text, unit_dep = unit
        pattern_unit = {
            'LEFT_ID': target_id,
            'REL_OP': '>',
            'RIGHT_ID': f'{unit_text}-{unit_dep}-{increment}',
            'RIGHT_ATTRS': {'ORTH': unit_text, 'DEP': unit_dep}
        }
        dependency_pattern.append(pattern_unit)
        increment += 1

    for i, unit in enumerate(parents):
        if i == 0:
            left_id = target_id
        else:
            prev_unit = parents[i-1]
            prev_text, prev_dep = prev_unit
            left_id = f'{prev_text}-{prev_dep}-{increment-1}'

        unit_text, unit_dep = unit
        pattern_unit = {
            'LEFT_ID': left_id,
            'REL_OP': '<',
            'RIGHT_ID': f'{unit_text}-{unit_dep}-{increment}',
            'RIGHT_ATTRS': {'ORTH': unit_text, 'DEP': unit_dep}
        }
        dependency_pattern.append(pattern_unit)
        increment += 1
    return dependency_pattern

def aye_aye(category, output, path, pickle_path, docs_path, development=False):
    print("Start Time:", datetime.now().strftime("%H:%M:%S"))
    seeds = util.fetch_lines(f'seeds/{category}.seed')
    lexicon = set(seeds)

    # List in order to measure precision later on
    generated_words = []

    # Check to see if NLP has been done already
    if isfile(docs_path):
        print('Importing Previously Parsed Documents')
        with open(docs_path, 'rb') as file: 
            docs = pickle.load(file)
    else:
        print('Reading CSV')
        texts = pd.read_csv(path, squeeze=True)
        texts.dropna(inplace=True)
        # Split sampledTexts into series of size <= MAX_TEXT_SIZE and finalize them into a list of docs
        split_texts = np.split(texts, [MAX_TEXT_SIZE*(i+1) for i in range(len(texts)//MAX_TEXT_SIZE)])
        joined_texts = ['. '.join(text) for text in split_texts]
        print('Performing Dependency Parsing')
        docs = [nlp(text) for text in joined_texts]
        with open(docs_path, 'wb') as file:
            pickle.dump(docs, file)
    
    # Check for already extraced patterns
    if isfile(pickle_path):
        with open(pickle_path, 'rb') as file: 
            extracted_patterns_dict = pickle.load(file) 
    else:
        extracted_patterns_dict = defaultdict(set)

    pattern_frequency_dict = defaultdict(int)

    if development:
        print('Pre Processing Done')

    for iteration in range(LOOPS):
        
        if development:
            print(f'Starting Loop {iteration+1} for category {category}'), 

        all_patterns = set()
        for doc in docs:
            all_patterns.update(pattern_generation.extract_patterns(doc, lexicon, LEFT_TOKENS, PARENT_TOKENS, RIGHT_TOKENS, MIN_PATTERN_COMPLEXITY, MAX_PATTERN_COMPLEXITY))

        if development:
            print('1: Patterns Extracted')
            progress = 0
            print(f'Total Patterns for Round {iteration+1}: {len(all_patterns)}')

        # Convert all pattern tuples to forms accepted by spaCy
        # then add them to the matcher
        # lastly, hash them so we can find the matches later.
        matcher = DependencyMatcher(nlp.vocab)
        hasher = {}
        new_patterns = [pattern for pattern in all_patterns if pattern not in extracted_patterns_dict]
        for pattern in new_patterns:
            dependency_pattern = convert_to_dependency_pattern(pattern)
            matcher.add(str(pattern), [dependency_pattern])
            hasher[nlp.vocab.strings.add(str(pattern))] = pattern

        if development:
            print('2: Completed Pattern Conversion')

        # Match on patterns and extract words
        progress = 0
        for doc in docs:
            matches = matcher(doc)
            for match in matches:
                match_id, match_tokens = match
                target_token = doc[match_tokens[0]]
                target_text = target_token.text
                pattern = hasher[match_id]
                extracted_patterns_dict[pattern].add(target_text)
            if development:
                progress += 1
                print(f'Made it {round((progress/len(docs))*100, 2)}% of the way via extraction')

        # Now extract the new patterns from the dict and add
        # the extracted text
        extracted_patterns = []
        for pattern in all_patterns:
            extracted_patterns.append((pattern, extracted_patterns_dict[pattern]))

        if development:
            print('3: Extraction Done on All Potential Patterns')

        scored_patterns = []
        for pattern_set in extracted_patterns:
            score = r_log_f(pattern_set[1], category, lexicon)
            scored_patterns.append((pattern_set[0], pattern_set[1], score))
        scored_patterns.sort(key=itemgetter(2), reverse=True)
        scored_patterns = [scored_pattern for scored_pattern in scored_patterns if pattern_frequency_dict[scored_pattern[0]] <= MAX_PATTERN_USES]

        if development:
            print('4: Patterns Scored and Trimmed')

        pattern_pool_size = PATTERN_POOL_INIT_SIZE + (PATTERN_POOL_SIZE_INCREASE * iteration)
        chosen_patterns = scored_patterns[:pattern_pool_size]
        for pattern in chosen_patterns:
            pattern_frequency_dict[pattern[0]] += 1

        candidate_words = set().union(*[chosen_pattern[1] for chosen_pattern in chosen_patterns])
        candidate_words = candidate_words.difference(lexicon)

        scored_words = []
        for word in candidate_words:
            candidate_word_patterns = [pattern for pattern in extracted_patterns if word in pattern[1]]
            score = avg_log(candidate_word_patterns, category, lexicon)
            scored_words.append((word, score))
        scored_words.sort(key=itemgetter(1), reverse=True)
        scored_words = [word for word in scored_words if word[0].isalpha()]
        chosen_words = [word[0] for word in scored_words[:WORDS_PER_ROUND]]
        generated_words += chosen_words
        lexicon = lexicon.union(set(chosen_words))

        if development:
            print('5: Words Scored and Trimmed')
            print(f'This Rounds Words: {chosen_words}')
            print()

        # Save pattern extractions for next time
        # Make sure it saves after every iteration
        with open(pickle_path, 'wb') as file: 
            pickle.dump(extracted_patterns_dict, file) 

    print('Extracted Words...')
    print(lexicon.difference(set(seeds)))
    print()

    # Write the output of generated words
    file = open(f'{output}{category}.txt','w')
    for word in generated_words:
        file.write(f'{word}\n')
    file.close()


    print("End Time:", datetime.now().strftime("%H:%M:%S"))