import pandas as pd
import numpy as np
import random
import math

class DecisionTree:
    _description = 'A universal Decision Tree, accounting for all types of data and predictions.'
    
    @property
    def description(self):
        return self._description

    def __init__(self, max_deapth = 30, min_gain = -0.00001, min_node_examples = 1, max_node_examples = 1000000000, feature_subset_volume = 1000000000, question_count = 50, seed = 610):
        self.max_node_examples = max_node_examples
        self.max_deapth = max_deapth
        self.min_gain = min_gain
        self.question_count = question_count
        self.seed = seed
        self.min_node_examples = min_node_examples
        self.feature_subset_volume = feature_subset_volume
        random.seed(seed)
        np.random.seed(seed)

    class _Question:
        def __init__(self, compared_value, comparator, column):
            self.compared_value, self.comparator, self.column = compared_value, comparator, column

    class _Node:
        def __init__(self):
            self._is_leaf = False

        @property
        def question(self):
            if self._is_leaf:
                raise KeyError("No question in a leaf node.")
            return self.__question
        @question.setter
        def question(self, que):
            self.__question = que
            self.__true_son, self.__false_son = DecisionTree._Node(), DecisionTree._Node()
        
        @property
        def true_son(self):
            return self.__true_son

        @property
        def false_son(self):
            return self.__false_son

        def makeLeaf(self, answer, failure_probability = np.nan):
            self._answer = answer
            self._is_leaf = True
            self.failure_probability = failure_probability

        def ask(self, data):
            if self._is_leaf:
                raise KeyError("No question in a leaf node.")
            return self.__question.comparator(data[self.__question.column], self.__question.compared_value)

    @staticmethod
    def _getFailureProbability(data):
        if len(data) == 0:
            return 1.0

        value_counts = data.value_counts().values
        all_sum = sum(value_counts)
        entropy = 0.0

        for i in value_counts:
            if i == 0:
                continue
            p = i / all_sum
            entropy -= p * math.log2(p)

        return entropy


    def _buildTree(self, v, data, prediction_target, column_types, unique_values, current_deapth):
        max_gain = self.min_gain - 1.0
        best_question = 1

        current_failure_probability = DecisionTree._getFailureProbability(prediction_target)

        if self.feature_subset_volume >= len(data.columns):
            chosen_features = data.columns
        else:
            chosen_features = list(data.columns)
            np.random.shuffle(chosen_features)
            chosen_features = chosen_features[self.feature_subset_volume:]

        for feature in chosen_features:
            if self.question_count < min(len(data), len(unique_values[feature])):
                values = [data.iloc[i][feature] for i in np.random.randint(0, len(data), self.question_count)]
            elif len(unique_values[feature]) > len(data):
                values = data[feature].values
            else: 
                values = unique_values[feature]

            for val in values: 
                current_question = lambda x, y : x == y
                if column_types[feature] == 'integer':
                    current_question = lambda x, y : float(x) >= float(y)

                prediction_true  = prediction_target[list(map(lambda x : current_question(x, val), data[feature]))]
                prediction_false = prediction_target[list(map(lambda x : not current_question(x, val), data[feature]))]
                if min(len(prediction_true), len(prediction_false)) < self.min_node_examples:
                    continue

                new_failure_probability = (DecisionTree._getFailureProbability(prediction_true) * len(prediction_true) + 
                                           DecisionTree._getFailureProbability(prediction_false) * len(prediction_false)) / len(data)
                
                if current_failure_probability - new_failure_probability > max_gain:
                    max_gain = current_failure_probability - new_failure_probability
                    best_question = DecisionTree._Question(val, current_question, feature)
        
        if max_gain < self.min_gain and len(data) <= self.max_node_examples or best_question == 1 or current_deapth == self.max_deapth and len(data) <= self.max_node_examples:
            result = prediction_target.value_counts().idxmax()
            v.makeLeaf(result, current_failure_probability)
            return

        question_results = np.array(list(map(lambda x :best_question.comparator(x, best_question.compared_value), data[best_question.column])))
        data_true  = data[question_results]
        data_false = data[question_results ^ True]
        prediction_true  = prediction_target[question_results]
        prediction_false = prediction_target[question_results ^ True]

        v.question = best_question
        self._buildTree(v.true_son, data_true, prediction_true, column_types, unique_values, current_deapth + 1)
        self._buildTree(v.false_son, data_false, prediction_false, column_types, unique_values, current_deapth + 1)

    @staticmethod
    def _get_column_types(data):
        column_types = {}

        for i in data.columns:
            column_types[i] = 'unknown'
            if type(data[i].iloc[0]) == type('abc'):
                column_types[i] = 'string'

            try:
                float(data[i].values[0])
                column_types[i] = 'integer'
            except:
                pass

        return column_types

    def fit(self, data, prediction_target, column_types = {}):
        if len(data) == 0:
            raise ValueError('The dataset is empty or contains missing values')

        if column_types == {}:
            column_types = DecisionTree._get_column_types(data)

        unique_values = dict([(feature, data[feature].unique()) for feature in data.columns])

        self._root = DecisionTree._Node()            

        self._buildTree(self._root, data, prediction_target, column_types, unique_values, 0)

    def fit_and_get_error(self, data, prediction_target, column_types = {}, splits = 10):
        if len(data) == 0:
            raise ValueError('The dataset is empty or contains missing values')

        if column_types == {}:
            column_types = DecisionTree._get_column_types(data)

        order = list(data.index)
        np.random.shuffle(order)

        block_size = len(data) // splits

        mistakes = 0

        while len(order) > 0:
            self.fit(data.drop(order[:block_size]), prediction_target.drop(order[:block_size]), column_types)
            for ind in order[:block_size]:
                if self.predict(data.loc[ind]) != prediction_target[ind]:
                    mistakes += 1
            order = order[block_size:]

        self.fit(data, prediction_target, column_types)
        return mistakes / len(data)

    def _getValue(self, v, data, result):
        if len(data) == 0:
            return

        if v._is_leaf:
            for i in data.index:
                result[i] = v._answer
            return

        question_results = np.array(list(map(v.ask, [data.iloc[i] for i in range(len(data))])))

        self._getValue(v.true_son, data[question_results], result)
        self._getValue(v.false_son, data[question_results ^ True], result)
    
    def predict(self, data):
        if len(data) == 0:
            raise ValueError('The dataset is empty or contains missing values')

        result = {}
        self._getValue(self._root, data, result) 

        return pd.Series(result)

class RandomForest(DecisionTree):
    def __init__(self, max_deapth = 30, min_gain = -0.00001, min_node_examples = 1, max_node_examples = 1000000000,
                 feature_subset_volume = 1000000000, question_count = 50, seed = 610, tree_count = 5, drop_data = 0):

        self.max_node_examples = max_node_examples
        self.max_deapth = max_deapth
        self.min_gain = min_gain
        self.question_count = question_count
        self.seed = seed
        self.min_node_examples = min_node_examples
        self.feature_subset_volume = feature_subset_volume
        self.tree_count = tree_count
        self.drop_data = drop_data
        random.seed(seed)
        np.random.seed(seed)

    def fit(self, data, prediction_target, column_types = {}):
        self.forest = [DecisionTree(max_deapth=self.max_deapth, min_gain=self.min_gain, min_node_examples=self.min_node_examples, max_node_examples=self.max_node_examples,
                                    feature_subset_volume=self.feature_subset_volume, question_count = self.question_count, seed=self.seed) for i in range(self.tree_count)]
        for tree in self.forest:
            if self.drop_data == 0:
                tree.fit(data, prediction_target, column_types)
            else:
                ind = list(data.index)
                np.random.shuffle(ind)
                tree.fit(data.drop(ind[:self.drop_data]), prediction_target.drop(ind[:self.drop_data]), column_types)

    def predict(self, data):
        indexes = list(data.index)
        col = pd.Series([{} for i in range(len(data))], indexes)
        for tree in self.forest:
            prediction = tree.predict(data)
            for i in indexes:
                col.loc[i][prediction.loc[i]] = col.loc[i].get(prediction.loc[i], 0) + 1

        for i in indexes:
            prediction.loc[i] = max(zip(col.loc[i].values(), col.loc[i].keys()))[1]
        return prediction
        

            
    