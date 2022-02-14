# sl_parsing_slp
This repository is to submit the assignment given in study group SLP.
I implemented an POS tagger and an dependency parser from scratch.

## Install dependencies
Via [poetry](https://python-poetry.org)
```bash
poetry install
```

## POS tagger
### Usage
Codes: [src/pos_tagger.py](https://github.com/hirobf10/sl_parsing_slp/blob/main/src/pos_tagger.py)
- Evaluation
```bash
poetry run python src/pos_tagger.py --mode=eval

# The path probabilities of 0 sentences are assigned as zeros.
# Precision: 0.8412918589260101, Recall: 0.786099163707078, f-score: 0.8018346446054213
```
- Inference
```bash
poetry run python src/pos_tagger.py --mode=pred

# Sentence:
# These include different variants of Lesk algorithm or most frequent sense algorithm .
# Prediction:
# ('These_DT include_VBP different_JJ variants_NN of_IN Lesk_DT algorithm_NN or_CC most_JJS frequent_NN sense_NN algorithm_NN ._.', 3.549059910553398e-46)
```

### Approach  
I adopted Hidden Marcov Model as the model of POS tagger.  


Brief explanation of my codes: 
- Count frequencies of tokens and bigrams in HMMTagger._get_count_dict
- Based on the frequencies, calculate transition prob. and emission prob. in HMMTagger._get_prob_dict
- Implement viterbi algorithm in HMMTagger._viterbi

### Experiments
- train dataset: [data/wiki-en-train.norm_pos](https://github.com/hirobf10/sl_parsing_slp/blob/main/data/wiki-en-train.norm_pos), test dataset: [data/wiki-en-test.norm_pos](https://github.com/hirobf10/sl_parsing_slp/blob/main/data/wiki-en-test.norm_pos)
- To deal with unknow words, I exploit a smoothing method for emission probability. I smoothed the probability with $\lambda$ (`lmd`) parameter [2], 0.995 in this experiment.
- In this assignment, I show the result of precision, recall and f-score.

### Results
#### Evaluation
| Precision | Recall | f-score |
|-----------|--------|---------|
| 0.841     | 0.786  | 0.801   |

#### Example
- A sample of labels the tagger assigned
```
Sentence:
It is based on the hypothesis that words used together in text are related to each other and that the relation can be observed in the definitions of the words and their senses .

Prediction:
It_PRP is_VBZ based_VBN on_IN the_DT hypothesis_NN that_IN words_NNS used_VBN together_RB in_IN text_NN are_VBP related_VBN to_TO each_DT other_JJ and_CC that_IN the_DT relation_NN can_MD be_VB observed_VBN in_IN the_DT definitions_NN of_IN the_DT words_NNS and_CC their_PRP$ senses_NNS ._.

Reference:
It_PRP is_VBZ based_VBN on_IN the_DT hypothesis_NN that_IN words_NNS used_VBN together_RB in_IN text_NN are_VBP related_JJ to_TO each_DT other_JJ and_CC that_IN the_DT relation_NN can_MD be_VB observed_VBN in_IN the_DT definitions_NNS of_IN the_DT words_NNS and_CC their_PRP$ senses_NNS ._.
```

## Dependency parser
### Usage
Codes: [src/dep_parser.py](https://github.com/hirobf10/sl_parsing_slp/blob/main/src/dep_parser.py)
- Evaluation
```bash
poetry run python src/dep_parser.py --mode=eval --seed=20 --topn=30 --lr=0.05

# UAS: 0.31685
```
- Inference
```bash
poetry run python src/dep_parser.py --mode=pred --seed=20 --topn=30 --lr=0.05

# Sentence:
# the backdrop to friday 's slide was markedly different from that of the october 1987 crash , fund managers argue .
# Prediction:
# [5, 20, 21, 5, 12, 12, 5, 20, 7, 20, 3, 20, 5, 5, 20, 12, 20, 20, 20, 0, 7]
# Reference:
# [2, 7, 2, 5, 6, 3, 20, 9, 7, 9, 10, 11, 16, 16, 16, 12, 20, 19, 20, 0, 20]
```

### Approach
I employed MST parser, one of graph-based dependency parser, for this assignment.


Explanation of my codes:
- [Dataset](https://github.com/hirobf10/sl_parsing_slp/blob/0e8004c88b465f7af035e5497b46ec8f09b355f4/src/dep_parser.py#L15) loads the dataset with CoNLL format
- [FeatureExtractor](https://github.com/hirobf10/sl_parsing_slp/blob/0e8004c88b465f7af035e5497b46ec8f09b355f4/src/dep_parser.py#L50) extracts features based on frequencies and `topn` parameter.
- [MSTParser](https://github.com/hirobf10/sl_parsing_slp/blob/0e8004c88b465f7af035e5497b46ec8f09b355f4/src/dep_parser.py#L166) contains the parsing function and the evaluating function.
    - MSTParser.parse parses a sentence based on features and trained parameters
    - MSTParser.evaluate evaluates test dataset, which should be an instance of Dataset class, and return averaged unlabeled attachment score.

### Experiments
- train dataset: [data/mstparser-en-train.dep](https://github.com/hirobf10/sl_parsing_slp/blob/main/data/mstparser-en-train.dep), test dataset: [data/mstparser-en-test.dep](https://github.com/hirobf10/sl_parsing_slp/blob/main/data/mstparser-en-test.dep)
- features I used: word form, pos, dependency, and combination of them between dependent and head. I discarded the feature with the frequency less than `topn`, 30 in this experiment.
- learning rate, which is to train weight parameters multiplied with the features, is 0.05
- I use unlabeled attachment score (UAS) to evaluate my parser.

### Results
#### Evaluation
UAS: 0.31685


Actually, I had completely misunderstood the way to update parameters and there are some faults in my implementation. Then, I re-implemented it in [src/dep_parser.py](https://github.com/hirobf10/sl_parsing_slp/blob/develop/src/dep_parser.py) (on develop branch) but there still seem to be bugs in my codes because UAS of the parser is quite bad like less than 0.1.  
At the moment, I am not sure where I had mistaken but I will reimplement it in the future.

#### Example
```
Sentence:
the year-to-date total was 12,006,883 tons , up 7.8 % from 11,141,711 tons a year earlier .

Prediction:
[16, 16, 4, 0, 16, 4, 16, 4, 16, 11, 5, 16, 4, 16, 11, 17, 4]

Reference:
[3, 3, 4, 0, 6, 4, 4, 4, 10, 8, 8, 13, 11, 15, 16, 11, 4]
```

## Reference
[1] https://web.stanford.edu/~jurafsky/slp3/  
[2] https://github.com/neubig/nlptutorial/blob/master/download/04-hmm/nlp-programming-en-04-hmm.pdf