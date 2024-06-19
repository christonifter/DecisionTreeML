import numpy as np
import pandas as pd
import math, random
import copy
from DecisionTree import DecisionTree

class ML_Data:

# initialize a ML_Data object containing data in a pandas dataframe
    def __init__(self, file_path:str, headers = []):
        self.file_path = file_path 
        self.data = self.load_csvdata(headers)

# Pre-processing methods
    def load_csvdata(self, headers: list) -> pd.DataFrame:
        '''Loads a csv file into pandas dataframe'''
        if headers:
            df = pd.read_csv(self.file_path, header = None)
            df.columns = headers
        else:
            df = pd.read_csv(self.file_path)
        self.data = df
        return df
    
    def replace_missing(self, missing_str: str, int_val = False, columns = []) -> pd.DataFrame:
        '''Replaces values indicated by missing_str with the column mean. Applies to the specified columns,
        indexed by number or name. Attempts to convert the columns to int or float'''
        out = self.data.copy()
        column_names = list(self.data.columns)
        if not columns:
            columns = list(range(self.data.shape[1]))
        column_names = list(self.data.columns)
        for column in columns:
            if isinstance(column, str):
                column_name = column
            else: 
                column_name = column_names[column]
            # find column means
            out[column_name] = pd.to_numeric(out[column_name], errors = 'ignore')
            column_array = pd.to_numeric(out[column_name], errors = 'coerce')
            column_mean = column_array[np.where(column_array != missing_str)[0]].mean()
            if int_val:
                column_mean = round(column_mean)
            # replace target string with column mean
            out[column_name] = out[column_name].replace(missing_str, column_mean)
            if int_val:
                out = out.astype({column_name: 'int'})
            else:
                out = out.astype({column_name: 'float'})
        return out
    
    def replace_categories(self, categorical_values = dict({'False': 0, 'True': 1}), columns = [], one_hot = False) -> pd.DataFrame:
        '''Meant to recode categorical entries (e.g. 'False', 'True', 'Monday', 'Tuesday') with numeric values.
        Alternatively applies one_hot coding if one_hot is True. Applies to the specified columns, indexed by number or name.'''
        out = self.data.copy()
        if not columns:
            columns = list(range(self.data.shape[1]))
        column_names = list(self.data.columns)
        for column in columns:
            if isinstance(column, str):
                column_name = column
            else: 
                column_name = column_names[column]
            # replace key with value
            for value in list(categorical_values.keys()):
                out[column_name] = out[column_name].replace(value, categorical_values[value])
            # one-hot encoding
            if one_hot:
                out = pd.concat([self.data, pd.get_dummies(self.data[column_name], prefix=column_name, dtype=int)], axis = 1)
                out.drop(column_name, axis = 1, inplace=True)
        return out
    
    def discretize(self, bins = 10, columns = [], quantile = False) -> pd.DataFrame:
        '''Discretizes the specified columns into bins by rounding each value to the bin centers. Bins are defined by
        the number of bins and equal-width (quantile = False) or equal-frequency (quantile = True).'''
        out = self.data.copy()
        if not columns:
            columns = list(range(self.data.shape[1]))
        column_names = list(self.data.columns)
        for column in columns:
            if isinstance(column, str):
                column_name = column
            else: 
                column_name = column_names[column]
            column_array = np.array(self.data[column_name])
            # create bins
            if quantile:
                bin_edges = np.quantile(column_array, np.linspace(0, 1, bins+1))
            else:
                bin_edges = np.linspace(column_array.min(), column_array.max(), bins+1)
            bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2
            # replace values with nearest bin_centers
            binned_column = pd.cut(column_array, bin_edges, labels = bin_centers)
            out[column_name] = binned_column
            out = out.astype({column_name: 'float'})
        return out

    def standardize(self, train: pd.DataFrame, test: pd.DataFrame, columns = []) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''Replaces values indicated by missing_str with the column mean. Applies to the specified columns,
        indexed by number or name. Attempts to convert the columns to int or float.'''
        out_train = train.copy()
        out_test = test.copy()

        if not columns:
            columns = list(range(train.shape[1]))
        column_names = list(train.columns)
        for column in columns:
            if isinstance(column, str):
                column_name = column
            else: 
                column_name = column_names[column]
            # z-score standardization
            train_column = np.array(train[column_name])
            train_norm = (train_column - train_column.mean())/train_column.std()
            test_column = np.array(test[column_name])
            test_norm = (test_column - train_column.mean())/train_column.std()
            out_train[column_name] = train_norm
            out_test[column_name] = test_norm
        return out_train, out_test

# Data partitioning methods    
    def kfold_crossvalid(self, k: int) -> tuple[list, list]:
        '''Shuffles the dataframe rows, then creates k training sets and k testing sets.
        The k testing sets will collectively sample every observation exactly once.
        Each training set is complementary to its paired testing set.'''
        tests = []
        trains = []
        nobs = self.data.shape[0]
        # shuffle rows
        shuffled_seq = np.random.permutation(nobs)
        df_shuf = self.data.iloc[shuffled_seq,]
        # define training and testing sets
        bounds = np.round(np.arange(k+1) * (nobs / k))
        for i in range(k):
            testbounds = np.arange(bounds[i], bounds[i+1], dtype=int)
            trainbounds = np.delete(np.arange(nobs), testbounds)
            test_df = df_shuf.iloc[testbounds, :]
            train_df = df_shuf.iloc[trainbounds, :]
            tests.append(test_df)
            trains.append(train_df)
        return trains, tests

    def crossvalid_kx2(self, k: int) -> tuple[list, list, list]:
        '''Shuffles the dataframe rows, then creates k training sets and k testing sets.
        The k testing sets will collectively sample every observation exactly once.
        Each training set is complementary to its paired testing set.'''
        tests = []
        train1s = []
        train2s = []
        nobs = self.data.shape[0]
        # define training and testing sets
        bounds = np.round(np.array([0, 1, (k+1)/2, k]) * (nobs / k))
        for i in range(k):
            # shuffle rows
            shuffled_seq = np.random.permutation(nobs)
            df_shuf = self.data.iloc[shuffled_seq,]
            testbounds = np.arange(bounds[0], bounds[1], dtype=int)
            train1bounds = np.arange(bounds[1], bounds[2], dtype=int)
            train2bounds = np.arange(bounds[2], bounds[3], dtype=int)
            test_df = df_shuf.iloc[testbounds, :]
            train1_df = df_shuf.iloc[train1bounds, :]
            train2_df = df_shuf.iloc[train2bounds, :]
            tests.append(test_df)
            train1s.append(train1_df)
            train2s.append(train2_df)
        return train1s, train2s, tests

# Prediction evaluation methods    
    def evaluate_classifier(self, observed, expected) -> float:
        '''Returns the classifier score, measured as the number of observations where the predicted label
        matches the observed label (0/1 loss).'''
        matches = np.where(observed == expected)[0].shape[0]
        n = observed.shape[0]
        return matches/n

    def evaluate_regression(self, observed, expected) -> tuple:
        '''Returns the regression mean squared error'''
        residuals = observed - expected
        sum_squared_error = (residuals**2).sum()
        n = observed.shape[0]
        mse = (sum_squared_error / n) ** .5
        r2 = (np.corrcoef(observed, expected)[0,1]) ** 2
        return mse, r2
    
    def null_model_classifier(self, train, test, labels) -> dict:
        '''Predicts the class of the test observations as the mode of the training labels'''
        model = dict()
        predict = np.ones((1, test.shape[0])) * labels.mode()[0]
        model['parameters'] = [labels.mode()[0]]
        model['predictions'] = predict
        return model
    
    def null_model_regression(self, train, test, labels) -> dict:
        '''Predicts the response of the test observations as the mean of the training response'''
        model = dict()
        predict = np.ones((1, test.shape[0])) * labels.mean()
        model['parameters'] = [labels.mean()]
        model['predictions'] = predict
        return model
    

    def kfold_regress(self, target, k, standardize_columns = []) -> tuple[list, list]:
        '''Reports the k-fold cross validation performance of the null model regression'''
        trains, tests = self.kfold_crossvalid(k)
        for i in range(k):
            trains[i], tests[i] = self.standardize(trains[i], tests[i], standardize_columns)
        performance = np.zeros((k,))
        for i in range(k):
            model = self.null_model_regression(trains[i], tests[i], trains[i][target])
            performance[i] = (self.evaluate_regression(np.array(tests[i][target]), model['predictions']))
        baseline = np.array(tests[i][target]).std()
        print(f'Baseline StDev: {baseline}')
        mse = performance.mean()
        print(f'Mean Squared Error: {mse}')
        return trains, tests

    def kx2_regress(self, target, k) -> tuple[list, list, list]:
        '''Reports the kx2 cross validation performance of the null model regression'''
        train1s, train2s, tests = self.crossvalid_kx2(k)
        performance = np.zeros((k*2,))
        for i in range(k):
            model1 = self.null_model_regression(train1s[i], tests[i], train1s[i][target])
            model2 = self.null_model_regression(train2s[i], tests[i], train2s[i][target])
            performance[2*i] = self.evaluate_regression(np.array(tests[i][target]), model1['predictions'])[0]
            performance[2*i+1] = self.evaluate_regression(np.array(tests[i][target]), model2['predictions'])[0]
        baseline = self.data[target].std()
        print(f'Baseline StDev: {baseline}')
        mse = performance.mean()
        print(f'Mean Squared Error: {mse}')
        return train1s, train2s, tests

    def kfold_classify(self, target, k, standardize_columns = []) -> tuple[list, list]:
        '''Reports the k-fold cross validation performance of the null model classifier'''
        trains, tests = self.kfold_crossvalid(10)
        for i in range(k):
            trains[i], tests[i] = self.standardize(trains[i], tests[i], standardize_columns)

        performance = np.zeros((k,))
        for i in range(k):
            model = self.null_model_classifier(trains[i], tests[i], trains[i][target])
            performance[i] = (self.evaluate_classifier(np.array(tests[i][target]), model['predictions']))
        classifier_score = performance.mean()
        print(f'Classifier Score: {classifier_score}')
        return trains, tests

    def kx2_classify(self, target, k) -> tuple[list, list]:
        '''Reports the kx2 cross validation performance of the null model classifier'''
        train1s, train2s, tests = self.crossvalid_kx2(k)
        performance = np.zeros((2*k,))
        for i in range(k):
            model1 = self.null_model_classifier(train1s[i], tests[i], train1s[i][target])
            model2 = self.null_model_classifier(train2s[i], tests[i], train2s[i][target])
            performance[2*i] = self.evaluate_classifier(np.array(tests[i][target]), model1['predictions'])
            performance[2*i+1] = self.evaluate_classifier(np.array(tests[i][target]), model2['predictions'])
        classifier_score = performance.mean()
        print(f'Classifier Score: {classifier_score}')
        return train1s, train2s, tests

    def nn_classifier(self, train_num_features, train_cat_features, train_labels, test_num_features, test_cat_features, test_labels, ks):
        '''Predicts classes of testing set, using k-nearest neighbors classification'''
        rows = np.array(test_num_features.index)
        accs = []
        for k in ks:
            label_pred = []
            for row in rows:
                obs_num = test_num_features.loc[row] # numeric features of test point
                obs_cat = test_cat_features.loc[row] # categorical features of test point
                neighb_num = train_num_features # numeric features of training set
                neighb_cat = train_cat_features # categorical features of training set
                nnrow = self.nn_search(obs_num, neighb_num, obs_cat, neighb_cat, train_labels, k)
                #plurality vote
                nn_labels = [train_labels[neighbor] for neighbor in nnrow]
                pred_label = max(nn_labels, key = nn_labels.count)
                label_pred.append(pred_label)
            temp = test_labels.copy()
            temp['prediction'] = label_pred
            accuracy = self.evaluate_classifier(test_labels, temp['prediction'])
            accs.append(accuracy)
        out = pd.DataFrame({'k': ks, 'accuracy': accs})
        return out

    def kmeans_regression(self, train_features, train_labels, test_features, test_labels, ks, spreads):
        '''Predicts values of testing set, using k-means regression'''
        mses = []
        r2s = []
        ktest = []
        spreadtest = []
        for k in ks:
            for spread in spreads:
                # cluster training data
                (cluster_df, cluster_means, error) = self.kmeans_cluster(train_features, k)
                # evaluate training data on rbf kernels at each cluster mean
                H = self.rbf_kernel(train_features, cluster_means, spread)
                W = np.linalg.pinv(H) @ train_labels
                # evaluate testing data on rbf kernels at each cluster mean
                Htest = self.rbf_kernel(test_features, cluster_means, spread)
                preds = Htest @ W
                mse, r2 = self.evaluate_regression(preds, test_labels)
                ktest.append(k)
                spreadtest.append(spread)
                mses.append(mse)
                r2s.append(r2)
        out = pd.DataFrame({'k': ktest, 'spread': spreadtest, 'mse': mses, 'r2': r2s})
        out2 = pd.DataFrame({'observed': test_labels, 'expected': preds})
        return out, out2
    
    def evaluate_tree(self, tree, test_features, test_labels, classify=True):
        '''
        Evaluates decision tree accuracy (classifier) or mean squared error (regressor) on test set
        '''
        nobs = test_labels.shape[0]
        preds = np.empty((nobs,))
        for i in range(test_labels.shape[0]):
            preds[i], err = tree.predict(test_features.iloc[i])
        if classify:
            out = self.evaluate_classifier(test_labels, preds)
        else:
            out, r2 = self.evaluate_regression(preds, test_labels)
        return out, preds

# Nonparametric prediction methods
    def kmeans_cluster(self, data, k):
        '''Clusters data into k clusters, using k-means'''
        temp = data.copy()
        nobs = data.shape[0]
        ndims = data.shape[1]
        points_per_dimension = np.max(np.array([math.ceil(k / nobs), 2]))
        gridpoints = self.make_grid(ndims, points_per_dimension)
        gridrows = random.sample(range(gridpoints.shape[0]), k)
        scale = np.array(data.std()).reshape(1, ndims)
        shift = np.array(data.mean()).reshape(1, ndims)
        cluster_means = pd.DataFrame(gridpoints[gridrows,:] * scale + shift, columns = data.columns) # k x ndims
        breakflag = True
        while breakflag:
            initial_means = cluster_means.copy()
            breakflag = False
            # Expectation (assign points to clusters)
            rows = np.array(data.index)
            cluster = []
            for row in rows:
                obs_num = data.loc[row]
                obs_cat = data.loc[row][[]]
                neighb_num = cluster_means
                neighb_cat = cluster_means[[]]
                nn = self.nn_search(obs_num, neighb_num, obs_cat, neighb_cat, [], k=1)[0]
                cluster.append(nn)
            temp['cluster'] = cluster
            # Maximization (find cluster means)
            distortion = 0
            for cluster_i in range(k):
                cluster_data = temp[temp['cluster'] == cluster_i]
                if cluster_data.shape[0] > 0:
                    cluster_means.iloc[cluster_i, :] = cluster_data[data.columns].mean()
                    distortion += np.linalg.norm(cluster_data[data.columns] - cluster_data[data.columns].mean())
            change = np.linalg.norm(cluster_means-initial_means)
            if change > 1:
                breakflag = True
        return temp, cluster_means, distortion

    def make_grid(self, ndims, ppd):
        '''Creates a grid of evenly spaced points along ndims dimensions, and ppd points per dimension'''
        ig = np.zeros((ppd**ndims, ndims))
        for row in range(1, ppd**ndims):
            num = row
            for col in range(ndims):
                rem = num % ppd
                ig[row,col] = rem
                num = (num - rem) / ppd
        ig = ig - (ppd-1)/2
        return ig
    
    def rbf_kernel(self, data, means, spread):
        '''Evaluates the data at kernels centered at given means'''
        nobs = data.shape[0]
        nrbfs = means.shape[0]
        H = np.zeros((nobs, nrbfs))
        for row in range(nobs):
            distance = ((means - data.iloc[row, :])**2).sum(axis=1)**.5
            H[row,:] = np.exp(-distance/spread)
        return H


    def cat_feature_distance(self, v1, v2, column, labels, p=2):
        '''Calculates distance for categorical features'''
        classes = labels.unique()
        c1den = np.where((column==v1))[0].shape[0]
        c2den = np.where((column==v2))[0].shape[0]
        distance = 0
        for tclass in classes:
            c1num = np.where((column==v1) & (labels==tclass))[0].shape[0]
            c2num = np.where((column==v2) & (labels==tclass))[0].shape[0]
            distance += (c1num/c1den - c2num/c2den) ** p
        return distance
    
    def VDM(self, row1, row2, features, labels, p=2):
        '''Calculates the value distance metric'''
        sum_d = 0
        for feature in features.columns:
            sum_d += self.cat_feature_distance(row1[feature], row2[feature], features[feature], labels, p)
        distance = sum_d ** (1/p)
        return distance 
    
    def nn_search(self, obs_num, neighb_num, obs_cat, neighb_cat, labels, k=1):
        '''Finds k nearest neighbors to an observation, based on distance of numeric and categorical features.
        The observation has numeric features: obs_num, and categorical features: obs_cat. The neighbors have
        numeric features: neighb_num, categorical features: neighb_cat, and labels: labels.
        obs_num and obs_cat must be 1-D vectors (Pandas series). neighb_num, neighb_cat, and labels must have 
        the same number of rows
        '''
        nneighbors = neighb_num.shape[0]
        # linear nearest neighbor search (this could be improved?)
        if nneighbors == 1:
            nn = np.array(neighb_num.index[0])
        else:
            num_distances = np.zeros(nneighbors)
            cat_distances = np.zeros(nneighbors)
            # L2 norm distance (numeric features)
            if neighb_num.shape[1] > 0:
                num_distances = ((neighb_num-obs_num)**2).sum(axis=1)**.5
            # VDM distance (categorical features)
            if neighb_cat.shape[1] > 0:
                c = 0
                neighbs = np.array(neighb_cat.index)
                for neighb_i in neighbs:
                    cat_distances[c] = self.VDM(obs_cat, neighb_cat.loc[neighb_i], neighb_cat, labels, p=2)
                    c += 1
            # combine distances
            distances = (num_distances **2 + cat_distances **2) ** .5
            temp = neighb_num.copy()
            temp['distance'] = distances
            temp.sort_values('distance', inplace=True)
            nn = np.array(temp.index[0:k])
        return nn       

## Decision Tree methods
    def Generate_Tree(self, data_cat, data_num, labels, theta=0, classify=True):
        '''
        Recursive generation of Decision Tree
        '''
        # features_are_identical = True when tree cannot be split (single row or all features identical between rows)
        features_are_identical_1 = ((data_cat.drop_duplicates().shape[0] == 1) and (data_num.shape[1] == 0))
        features_are_identical_2 = ((data_num.drop_duplicates().shape[0] == 1) and (data_cat.shape[1] == 0))
        features_are_identical_3 = (data_cat.drop_duplicates().shape[0] + data_num.drop_duplicates().shape[0] == 2)
        features_are_identical = features_are_identical_1 or features_are_identical_2 or features_are_identical_3
        # Recursion base case: generate leaf node
        if classify: # classification
            if (self.entropy(labels) < theta) or features_are_identical:
                out = self.get_majority_class(labels)
                majority_class = out[0]
                p_class = out[1]
                return DecisionTree(feature_name = majority_class, keys = p_class, train_labels = labels)
        else: # regression
            if (self.variance(labels) < theta) or features_are_identical:
                estimate = labels.mean()
                error = self.variance(labels)
                return DecisionTree(feature_name = estimate, keys = error, train_labels = labels)
        # General case: generate node with children
        best_feature, data_cat_splits, data_num_splits, labels_splits, split_criteria  = self.get_best_split(data_cat, data_num, labels, classify)
        children = []
        n_children = len(labels_splits)
        
        for i in range(n_children):
            children.append(self.Generate_Tree(data_cat_splits[i], data_num_splits[i], \
                                                labels_splits[i], theta, classify))
        return DecisionTree(feature_name = best_feature, keys = split_criteria, children = children, train_labels = labels)

    def get_best_split(self, data_cat, data_num, labels, classify):
        '''
        Splits data on feature and threshold that maximizes gain ratio (classification) 
        or minimizes error (regression)
        '''
        nobs = labels.shape[0] # number of observations
        max_split_metric = -1E10 # metric to maximize: gain ratio (classification) or -1*error (regression)
        features_cat = list(data_cat.keys())
        features_num = list(data_num.keys())
        for feature_cat in features_cat:
            # split data by levels (observed variates of categorical feature)
            levels = list(data_cat[feature_cat].unique())
            data_cat_splits = []
            data_num_splits = []
            labels_splits = []
            for level in levels:
                level_rows = data_cat[feature_cat] == level
                data_cat_splits.append(data_cat[level_rows])
                data_num_splits.append(data_num[level_rows])
                labels_splits.append(labels[level_rows])
            # evaluate split metric
            e = self.split_feature(labels, labels_splits, classify)
            if e > max_split_metric:
                max_split_metric = e
                bestf = feature_cat
                data_cat_out = data_cat_splits
                data_num_out = data_num_splits
                labels_out = labels_splits
                split_criteria = list(data_cat[bestf].unique())
        for feature_num in features_num:
            # split data by thresholds (midpoint between consecutive sorted unique values)
            nums = np.unique(data_num[feature_num])
            thresholds = (nums[0:-1] + nums[1:]) /2
            for threshold in thresholds:
                rows_1 = data_num[feature_num] < threshold
                rows_2 = data_num[feature_num] > threshold
                data_cat_splits = [data_cat[rows_1], data_cat[rows_2]]
                data_num_splits = [data_num[rows_1], data_num[rows_2]]
                labels_splits = [labels[rows_1], labels[rows_2]]
                # evaluate split metric
                e = self.split_feature(labels, labels_splits, classify)
                if e > max_split_metric:
                    max_split_metric = e
                    bestf = feature_num
                    data_cat_out = data_cat_splits
                    data_num_out = data_num_splits
                    labels_out = labels_splits
                    split_criteria = [threshold]
        return bestf, data_cat_out, data_num_out, labels_out, split_criteria

    def split_feature(self, labels, label_splits, classify):
        '''
        Measures gain ratio or mean sum squared error across splitted data
        '''
        nobs = labels.shape[0]
        if len(label_splits) == 1: # unable to split (data contains single feature level), prioritize low 
            split_metric = -1E10
        else:
            if classify:
                E_pi = 0
                IV = 0
                for label_split in label_splits:
                    p_level = len(label_split)/nobs #sample probability that feature = level
                    E_pi = E_pi + p_level * self.entropy(label_split)
                    IV = IV - p_level * math.log(p_level)
                split_metric = (self.entropy(labels) - E_pi) / IV # gain ratio
            else:
                split_metric = 0
                for label_split in label_splits:
                    mu = label_split.mean()
                    sigma_2 = 1/nobs*(label_split - mu)**2
                    split_metric = split_metric - sigma_2.sum() # negative mean sum squared error
        return split_metric

    def get_majority_class(self, labels):
        majority_class = labels.value_counts().keys()[0]
        p_class = list(labels.value_counts())[0]/labels.shape[0]
        return majority_class, p_class
        
    def entropy(self, labels):
        nobs = labels.shape[0]
        classes = list(pd.unique(labels))
        I = 0
        for class_i in classes:
            p_class = sum(labels == class_i)/nobs
            I = I - p_class * math.log(p_class)
        return I

    def variance(self, labels):
        mu = labels.mean()
        nobs = labels.shape[0]
        var = 1/nobs*(labels - mu)**2
        return var.sum()

    def iter_prune(self, tree, p_features, p_labels, classify=True):
        '''
        Iteratively repeats prune(), until all parent nodes do not exceed the base performance. 
        '''
        breakflag = False
        iterations = 0
        while breakflag == False:
            iterations += 1
            out, node = self.prune(tree, p_features, p_labels, classify)
            if out == 'no change':
                breakflag = True
            else:
                tree = out
            if iterations > 100:
                breakflag = True
        return tree

    def prune(self, tree, p_features, p_labels, classify):
        '''
        Iterates through the parent nodes of the full tree, and tests the performance of a tree pruned at the iterated node.
        Once a pruned tree with higher performance than the full tree is found, the pruned tree is returned without completing the iterations.   
        '''
        base_metric, preds = self.evaluate_tree(tree, p_features, p_labels, classify)
        parent_nodes = self.vertices(tree, location=[], node_list=[])
        if classify:
            sign = 1
        else:
            sign = -1
        code_list = parent_nodes[1:]
        # iterate through parent nodes
        for parent_node in code_list:
            dummy_tree = copy.deepcopy(tree)
            pruned_tree = self.replace_node(dummy_tree, parent_node, classify) # pruned version of current tree
            pruned_metric, preds = self.evaluate_tree(pruned_tree, p_features, p_labels, classify) # performance of pruned version
            if (pruned_metric * sign) > (base_metric * sign):
                return pruned_tree, parent_node
        return 'no change', []

    def replace_node(self, tree, code, classify=True):
        '''
        Prunes a node (replaces the node with a leaf node). The node is navigated to by following the root's children
        in the sequence specified by parameter 'code'.
        '''
        nodestring = 'tree'
        for children_i in code:
            nodestring += f'.children[{children_i}]'
        exec_string = f'{nodestring} = self.make_leaf({nodestring}.train_labels, classify)'
        exec(exec_string) # executes: tree.children[a].children[b]... = self.makeleaf(...)
        return tree

    def vertices(self, tree, location=[], node_list=[]):
        '''
        Returns a list of all parent (non-leaf) nodes in the tree. Each element contains the node location, which can be navigated to 
        by following the children of the root according the sequence.
        '''
        if len(tree.children) == 0:
            return []
        else:
            node_list.append(location)
            for i in range(len(tree.children)):
                self.vertices(tree.children[i], location + [i], node_list)
            return node_list

    def make_leaf(self, labels, classify):
        '''
        Converts a non-leaf node to a leaf node by evaluating the mean or majority class of the node training set labels.
        '''
        if classify: # classification
            out = self.get_majority_class(labels)
            majority_class = out[0]
            p_class = out[1]
            return DecisionTree(feature_name = majority_class, keys = p_class)
        else: # regression
            estimate = labels.mean()
            error = self.variance(labels)
            return DecisionTree(feature_name = estimate, keys = error)


# Data reduction methods        
    def edited_nn(self, train_features_num, train_features_cat, train_labels, margin=0.1):
        '''Produces a condensed train of size k from training set of size n, where 0 < k <= n,
        using Stepwise-Backward Selection.'''
        breakflag = True
        subsetrows = np.array(train_features_num.index)
        subset_features_num = train_features_num.copy()
        subset_features_cat = train_features_cat.copy()
        subset_labels = train_labels.copy()

        while breakflag:
            breakflag = False
            subsetrows = np.array(subset_features_num.index)
            remove_rows = np.array([])
            for row in subsetrows:
                obs_num = subset_features_num.loc[row]
                obs_cat = subset_features_cat.loc[row]
                neighb_num = subset_features_num.drop(index=row)
                neighb_cat = subset_features_cat.drop(index=row)
                labels = train_labels.drop(index=row)
                nnrow = int(self.nn_search(obs_num, neighb_num, obs_cat, neighb_cat, labels, k=1))
                if abs(train_labels.loc[nnrow] - train_labels.loc[row]) > margin:
                    remove_rows = np.append(remove_rows, [row])
                    breakflag = True
            # batch removal
            subset_features_num.drop(remove_rows, inplace=True)
            subset_features_cat.drop(remove_rows, inplace=True)
            subset_labels.drop(remove_rows, inplace=True)
        return subset_features_num, subset_features_cat, subset_labels, subsetrows

    def condensed_nn(self, train_features_num, train_features_cat, train_labels, margin=0.1):
        '''Produces a condensed train of size k from training set of size n, where 0 < k <= n,
        using Stepwise-Forward Selection.'''
        allrows = np.array(train_features_num.index)
        nobs = allrows.shape[0]
        shuffled_seq = np.random.permutation(nobs)
        shuffled_rows = allrows[shuffled_seq]
        subset_rows = np.array([shuffled_rows[0]])
        breakflag = True
        while breakflag and (subset_rows.shape[0] < nobs):
            breakflag = False
            shuffled_seq = np.random.permutation(nobs)
            shuffled_rows = allrows[shuffled_seq]
            for row in shuffled_rows:
                obs_num = train_features_num.loc[row]
                obs_cat = train_features_cat.loc[row]
                neighb_num = train_features_num.loc[subset_rows]
                neighb_cat = train_features_cat.loc[subset_rows]
                labels = train_labels.loc[subset_rows]
                nnrow = int(self.nn_search(obs_num, neighb_num, obs_cat, neighb_cat, labels, k=1))
                cond1 = abs(train_labels.loc[nnrow] - train_labels.loc[row]) > margin
                cond2 = row not in subset_rows 
                if cond1 and cond2:
                    subset_rows = np.append(subset_rows, [row])
                    breakflag = True
            shuffled_seq = np.random.permutation(nobs)
            shuffled_rows = allrows[shuffled_seq]
        
        return train_features_num.loc[subset_rows], train_features_cat.loc[subset_rows], train_labels.loc[subset_rows], subset_rows