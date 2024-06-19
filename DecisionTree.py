import math
class DecisionTree:
    def __init__(self, feature_name = [], keys = [], children = [], train_labels = []):
        '''
        Constructs a Tree object.
        children: pointer to next linked node (towards the leafs).
        '''
        self.feature_name = feature_name
        self.keys = keys
        self.children = children
        self.train_labels = train_labels

    def predict(self, test_sample):
        '''
        Recursive traversal of Decision Tree 
        '''
        # Base case (leaf node): return prediction, error/probability
        if len(self.children) == 0:
            return self.feature_name, self.keys
        # General case: choose child node, based on feature and criteria
        else:
            test_value = test_sample[self.feature_name]
            if len(self.keys) == 1: # numerical feature
                if test_value < self.keys[0]:
                    out = self.children[0].predict(test_sample)
                else:
                    out = self.children[1].predict(test_sample)
            else: # categorical feature
                if test_value in self.keys:
                    test_index = self.keys.index(test_value)
                else:
                    test_index = 0
                out = self.children[test_index].predict(test_sample)
            return out
        
