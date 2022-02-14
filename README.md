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

# The path probabilities of 154 sentences are assigned as zeros.
# Precision: 0.5632051282051282, Recall: 0.04469060798621794, f-score: 0.08179061260804937, Average of PPL: 405.49789665762137
```
- Inference
```bash
poetry run python src/pos_tagger.py --mode=pred --seed=28

# Sentence:
# One solution some researchers have used is to choose a particular dictionary , and just use its set of senses .
# Prediction:
# ('One_CD solution_NN some_DT researchers_NNS have_VBP used_VBN is_VBZ to_TO choose_VB a_DT particular_JJ dictionary_NN ,_, and_CC just_RB use_VB its_PRP$ set_NN of_IN senses_NNS ._.', 1.0115399000049753e-52)
```

### Approach  
I adopted Hidden Marcov Model as the model of POS tagger.  


Brief explanation of my codes: 
- Count frequencies of tokens and bigrams in HMMTagger._get_count_dict
- Based on the frequencies, calculate transition prob. and emission prob. in HMMTagger._get_prob_dict
- Implement viterbi algorithm in HMMTagger._viterbi

### Experiments
- train dataset: [data/wiki-en-train.norm_pos](https://github.com/hirobf10/sl_parsing_slp/blob/main/data/wiki-en-train.norm_pos), test dataset: [data/wiki-en-test.norm_pos]((https://github.com/hirobf10/sl_parsing_slp/blob/main/data/wiki-en-test.norm_pos))
- No hyper parameters.
- No special implementation for unknown tokens. This means that if all bigrams in a sentence are not seen in train dataset, the tagger cannot calculate the probabily of the sentence and assign any label.
- In this assignment, I show precision, recall, f-score and perplexity.

### Results
#### Evaluation
The model cannot calculate the probability of 154 sentences of the test dataset.
In the following table, I report each metrics for the sentences where the model can calculate the probability.
| Precision | Recall | f-score | PPL     |
|-----------|--------|---------|---------|
| 0.563     | 0.044  | 0.081   | 405.497 |

#### Example
- One sample that the tagger seems to assign each label correctly
```txt
Part-of-speech_JJ tagging_NN In_IN any_DT real_JJ test_NN ,_, part-of-speech_JJ tagging_NN and_CC sense_NN tagging_NN are_VBP very_RB closely_RB related_JJ with_IN each_DT potentially_RB making_VBG constraints_NNS to_TO the_DT other_JJ ._.
```
- One that the tagger cannot assign any label
```txt
These_None include_None different_None variants_None of_None Lesk_None algorithm_None or_None most_None frequent_None sense_None algorithm_None ._None
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
```txt
Sentence:
the backdrop to friday 's slide was markedly different from that of the october 1987 crash , fund managers argue .
Prediction:
[5, 20, 21, 5, 12, 12, 5, 20, 7, 20, 3, 20, 5, 5, 20, 12, 20, 20, 20, 0, 7]
Reference:
[2, 7, 2, 5, 6, 3, 20, 9, 7, 9, 10, 11, 16, 16, 16, 12, 20, 19, 20, 0, 20]
```

## Reference
[1] https://web.stanford.edu/~jurafsky/slp3/