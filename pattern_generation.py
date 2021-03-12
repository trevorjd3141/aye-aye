# Takes a maximized pattern template and returns a set
# of all subsets of it
def expand_patterns(pattern_template, min_pattern_complexity, max_pattern_complexity):
    patterns = set()
    left_units, parent_units = pattern_template
    for i in range(len(left_units)+1):
        for j in range(len(parent_units)+1):
            if i+j >= min_pattern_complexity and i+j <= max_pattern_complexity:
                patterns.add((left_units[:i], parent_units[:j]))
    return patterns

# Patterns are in the form
# (unit) = (word, dep)
# (tuple) = (left units, parent units)
def extract_patterns(doc, lexicon, max_up=2, max_left=2, min_pattern_complexity=2, max_pattern_complexity=3):
    patterns = set()
    for token in doc:
        if token.text not in lexicon:
            continue
        left_tokens = [token for token in token.lefts if token.dep_ != 'compound']
        left_units = tuple([(left_token.text, left_token.dep_) for left_token in left_tokens[-max_left:]])
        ancestor_list = [token] + list(token.ancestors)
        parent_units = tuple([(parent_token.head.text, parent_token.dep_) for parent_token in ancestor_list[:max_up] if parent_token.dep_ != 'ROOT'])
        pattern_template = (left_units, parent_units)
        patterns.update(expand_patterns(pattern_template, min_pattern_complexity, max_pattern_complexity))
    return patterns