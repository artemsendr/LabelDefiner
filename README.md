# LabelDefiner
## Classification of phrases with typos and additions
### Description

Tool helps to search relevant items in long list and assign relevant class to them. The tool takes any 
short list of categories and assigns them to items in a long list which contains some typos, additions
in different formats of spelling, etc. If an item in the long list is not close enough to any 
item in the short list, it classified as OTHER.

This problem might be solved with a set of regular expressions, but setting up and support of them takes 
a lot of human resources. That's why the problem was solved using some data science instruments: 
NLP for comparing potential labels and input, and logistic regression for making a decision 
if the label is close enough to the actual input to be the category.

### Installation and first use

1) Clone the project
2) Install dependencies file `pip install -r requirements.txt`
3) Although the project contains pickeled regression model files it is needed to generate new ones
on your data due to these files are generated on my particular dataset of series and movies names.
4) Correct LabelDefiner.py file `main()` function to provide correct input flow, suggested labels,
regular expressions and to manage output flow.

### Usage
Project contains the class for classification short phrases to number of labels fitted or OTHER if there is not enough confidence that
    it is one of that labels. So to use it you need:

1) Create LabelDefiner class object with appropriate pipelined classification model which is implemented as function of three parameters -
        normalized Indel distance (syntax), embedding cosine similarity, and normalized partial spelling distance
        and returns probability of being not OTHER class for a phrase having these parameters with a given label.
_Also look on **Installation and first use** note **3**!_
2) Provide suggested labels with `fit(input_labels)` function
3) Get your classification results with `predict(input_flow, extended_output=False, regex=None)` function. 
If detalization of intermediate prediction parameters needed use `extended_output=True`.
