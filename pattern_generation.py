from itertools import product

# Takes a maximized pattern template and returns a set
# of all subsets of it
def expand_patterns(pattern_template, min_pattern_complexity, max_pattern_complexity):
    patterns = set()
    target_unit, left_units, parent_units, right_units = pattern_template

    # Get all possible combinations of patterns from the pattern template
    # itertools creates a cartesian product between the length of left, parent, and right
    for i, j, k in product(range(len(left_units)+1), range(len(parent_units)+1), range(len(right_units)+1)):
        if i+j+k >= min_pattern_complexity and i+j+k <= max_pattern_complexity:
            patterns.add((target_unit, left_units[:i], parent_units[:j], right_units[:k]))
    return patterns

# Patterns are in the form
# (unit) = (word, dep)
# (tuple) = (target unit, left units, parent units, right_units)
def extract_patterns(doc, lexicon, max_left=2, max_up=2, max_right=1, min_pattern_complexity=2, max_pattern_complexity=3):
    patterns = set()
    for token in doc:
        if token.text not in lexicon:
            continue
        target_unit = (token.text, token.dep_)
        left_tokens = [token for token in token.lefts if token.dep_ != 'det' and token.dep_ != 'punct']
        left_units = tuple([(left_token.text, left_token.dep_) for left_token in left_tokens[-max_left:]])
        parent_units = tuple([(parent_token.text, parent_token.dep_) for parent_token in list(token.ancestors)[:max_up] if parent_token.head.dep_ != 'ROOT'])
        right_token = [token for token in token.rights if token.dep_ != 'det' and token.dep_ != 'punct']
        right_units = tuple([(right_token.text, right_token.dep_) for right_token in right_token[:max_right]])
        pattern_template = (target_unit, left_units, parent_units, right_units)
        patterns.update(expand_patterns(pattern_template, min_pattern_complexity, max_pattern_complexity))
    return patterns