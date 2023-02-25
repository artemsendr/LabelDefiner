from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch
import rapidfuzz
from pickle import load


def classification_model_creation(model_file, preprocessing_file):
    """
    Loads classification and preprocessing models from specified pickle files and returns a pipelined model
    which making preprocessing and than classification
    :param model_file: pkl file with classification
    :param preprocessing_file: pkl file with preprocessing
    :return: classification model with preprocessing of arguments
    """
    model_clf = load(open(model_file, 'rb'))
    scaler = load(open(preprocessing_file, 'rb'))

    def classification_model(args):
        return model_clf.predict_proba(scaler.transform(args))

    return classification_model


class LabelDefiner:
    """
    Class for classification short phrases to number of labels fitted or OTHER if there is not enough confidence that
    it is one of that labels.
    Methods
    -------

    fit(input_labels)
        Assign suggested labels to the class.
    predict(input_flow, extended_output=False, regex=None)
        Predict class labels for each of input_flow phrases as either to one of input_labels or to OTHER.
    choose_class(channel, number_of_top=3)
        Defines top-N the most close categories (of categories fitted) to a given channel-name and it's distances.
        Distance is defined by _sintax_distance function.
    cosine_similarity_calc(input_flow)
        Calculates input series and fitted labels embeddings cosine similarity
    """

    def __init__(self, decision_maker, embedding_model=None, syntax_distance=None):
        """
        Creates LabelDefiner object with null labels using  parameters provided
        :param decision_maker: classification model which is implemented as function of three parameters -
        normalized Indel distance (syntax), embedding cosine similarity, and normalized partial spelling distance
        and returns probability of being not OTHER class for a phrase having these parameters with a given label.
        I.e. is there enough confidence that this label is relevant one
        :param embedding_model: model to use for embeddings calculation for labels and an input. If None than
                                BERT-based SentenceTransformers model distiluse-base-multilingual-cased-v2
        :param syntax_distance: function of two string to be used to measure distance between two strings (must return
                                numerical). If None than self.__reverse_fuzz_ratio(s1, s2)
        """
        self._input_labels = None
        self._decisionMaker = decision_maker

        if embedding_model:
            self._embedding_model = syntax_distance
        else:
            self._embedding_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

        if syntax_distance:
            self._sintax_distance = syntax_distance
        else:
            self._sintax_distance = self.__reverse_fuzz_ratio
        self._labels_embeddings = None

    @staticmethod
    def __reverse_fuzz_ratio(word1, word2):
        """returns distance between two input strings (reverse of rapidfuzz.fuzz.ratio)"""
        return 100 - rapidfuzz.fuzz.ratio(word1, word2)

    @staticmethod
    def __reverse_partial_ratio(s1, s2):
        """returns partial distance between two input strings (reverse of rapidfuzz.fuzz.partial_ratio)"""
        return 100 - rapidfuzz.fuzz.partial_ratio(s1, s2)

    def choose_class(self, channel, number_of_top=3):
        """
        Defines top-N the most close categories (of categories fitted) to a given channel-name and it's distances.
        Distance is defined by _sintax_distance function.
        :param channel: string
        :param number_of_top: number of top returned
        :return: number_of_top closest labels from channel and it's distances in pd.Series (names, distances) with names
                    set as index and distance as values
        """
        df = pd.DataFrame({'name': self._input_labels})
        df['dist'] = df.name.apply(lambda x: self._sintax_distance(x, channel))
        ind_top = np.argpartition(df.dist, number_of_top)[:number_of_top]
        ind_top_sorted = ind_top[np.argsort(df.dist[ind_top])]
        name_top = df.name[ind_top_sorted]
        dist_top = df.dist[ind_top_sorted]

        return pd.Series(data=dist_top.tolist(), index=name_top.tolist())

    def cosine_similarity_calc(self, input_flow):
        """
        Calculates input series and fitted labels embeddings cosine similarity
        Embeddings obtained with _embedding_model
        :param input_flow: pd.Series of phrases to be categorized
        :return: DataFrame of cosine similarities between input_flow and labels (column names are labels, row are
        input_flow)
        """
        input_emb = self._embedding_model.encode(input_flow.tolist(), convert_to_tensor=True).transpose(1, 0)

        # product is cosine similarity (scalar_product between each pair of normalized embedding vectors. We need to normalize along 512-dim axis.)
        product = torch.mm(self._labels_embeddings / self._labels_embeddings.norm(dim=1)[:, None],
                           input_emb / input_emb.norm(dim=0)[None, :])
        df_product = pd.DataFrame(product.cpu().numpy()).T
        dict_n = dict(zip([x for x in range(self._input_labels.size)], self._input_labels.tolist()))
        return df_product.rename(columns=dict_n)

    def fit(self, input_labels):
        """
        Assigning of suggested labels to the class object
        Calculates embeddings of all suggested labels and save it in class
        :param input_labels: pandas Series with suggested labels
        :return: void
        """

        # TODO: why not accept a list? you use a list anyway
        if not isinstance(input_labels, pd.Series):
            raise TypeError("wrong type of input. Should be pd.Series")
        names_list = input_labels.tolist()
        self._input_labels = input_labels
        self._labels_embeddings = self._embedding_model.encode(names_list, convert_to_tensor=True)

    def predict(self, input_flow, extended_output=False, regex=None):
        """
        labeling every row in input either to one of suggested classes (from self.input_labels) or to OTHER if
        model (from self._decisionMaker) classifies that input row belongs to none ot them
        :param input_flow: pd.Series input of any strings which need to be categorised
        :param extended_output: format of output. If "False" than just predicted labels. If "True" than DataFrame
        with the next fields:
            input - input_flow after regex (if any)
            top-i cat - i-th closer label by sintax_distance() distance to input
            top-i dist - sintax_distance() from top-i cat to input
            cos_sim_i_cat - cosine similarity of embeddings (defined by embedding_model()) between i-category and input
            top-i dist _pat - normalized partial spelling distance between i-category and input
            score_i - probability of being actual label for i-category (defined by  self._decisionMaker)
            class_ - result of classification (label)
        :param regex: regular expressions collection to be applied to all input_flow rows before calculating any
        parameters for decision model
        :return: pd.Series with labels in same order as input.
        """
        if not isinstance(input_flow, pd.Series):
            raise TypeError("wrong type of input. Should be pd.Series")
        if regex:
            for r in regex:
                input_flow = input_flow.str.replace(r, '', regex=True)

        df_product = self.cosine_similarity_calc(input_flow)

        # make dataframe which will contain input, intermediate parameters and results
        _all_data = pd.DataFrame({'input': input_flow})
        # assigning for each input row the top-3 closest by sintax_distance labels
        _all_data[["top-" + str(x) + " cat" for x in range(3)] + ["top-" + str(x) + " dist" for x in range(3)]] = \
            _all_data.input.astype('str').apply(
            lambda x: (m := self.choose_class(x), pd.Series(data=np.append(m.index.values, m.values, axis=0)))[
                -1])

        # assigning of cosine similarity between these top-3 labels and row input itself
        _all_data[['cos_sim_0_cat', 'cos_sim_1_cat', 'cos_sim_2_cat']] = pd.Series(np.zeros(3))
        for r in _all_data.index:
            for i in range(3):
                _all_data.loc[r, 'cos_sim_' + str(i) + '_cat'] = df_product.loc[
                    r, _all_data.loc[r, "top-" + str(i) + " cat"]]

        # calculating partial spelling distance between these top-3 candidate label and input.
        for i in range(3):
            _all_data["top-" + str(i) + " dist _pat"] = _all_data.apply(
                lambda x: self.__reverse_partial_ratio(str(x['input']), x["top-" + str(i) + " cat"]), axis=1)

        # calculate score for each of top using their 3 parameters(distance, cosine_similarity, partial distance)
        # with classification model built in separate code
        for i in range(3):
            _all_data['score_' + str(i)] = self._decisionMaker(_all_data[['top-' + str(i) + ' dist',
                                                                                    'cos_sim_' + str(i) + '_cat',
                                                                                    'top-' + str(i) + ' dist _pat']])\
                [:, 1]

        # assigning class on which score calculated above is highest and above 0.5 (if below class is OTHER)
        _all_data['class_'] = np.nan
        for i in range(3):
            _all_data.loc[_all_data['class_'].isna(), 'class_'] = _all_data.loc[
                _all_data['class_'].isna(), 'top-' + str(i) + ' cat'].where(
                (_all_data[['score_0', 'score_1', 'score_2']].idxmax(axis=1) == 'score_' + str(i)) & (
                        _all_data['score_' + str(i)] > 0.5))
        _all_data.loc[_all_data['class_'].isna(), 'class_'] = 'OTHER'
        if extended_output:
            return _all_data
        else:
            return _all_data['class_']


def main():
    classification_model = classification_model_creation('calssifier_2.pkl', 'scaler_2.pkl')
    labler = LabelDefiner(classification_model)
    regex = ["S\d{1,3}\s{0,3}E\d{1,3}"]
    input_labels = pd.Series(["apple", 'yandex', 'Google',
                              'Amazon'])
    input_flow = pd.Series(["apple", 'yango', 'Google',
                            'AmizonS01E02'])
    labler.fit(input_labels)

    out = labler.predict(input_flow, regex=regex)
    out_noregex = labler.predict(input_flow, extended_output=True)

    print(out, '\n', out_noregex)


main()
