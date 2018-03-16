import numpy as np
import random
import copy
import math


def train_error(dataset):
    '''
        TODO:
        Calculate the train error of the subdataset and return it.
        For a dataset with two classes:
        C(p) = min{p, 1-p}
    '''
    if len(dataset) == 0:
        return 0
    count = 0.0
    for i in range(len(dataset)):
        if dataset[i][0] == 0:
            count = count + 1.0
    p=count/len(dataset)
    if p == 0 or p == 1:
        return 0
    else:
      return min(p,1-p)


def entropy(dataset):
    '''
        TODO:
        Calculate the entropy of the subdataset and return it.
        This function is used to calculate the entropy for a dataset with 2 classes.
        Mathematically, this function return:
        C(p) = -p*log(p) - (1-p)log(1-p)
    '''
    if len(dataset) == 0:
        return 0
    count = 0.0
    for i in range(len(dataset)):
        if dataset[i][0] == 0:
            count = count + 1.0
    p=count/len(dataset)
    if p == 0 or p == 1:
        return 0
    else:
        return -p*(math.log2(p))-(1-p)*math.log2(1-p)


def gini_index(dataset):
    '''
        TODO:
        Calculate the gini index of the subdataset and return it.
        For dataset with 2 classes:
        C(p) = 2*p*(1-p)
    '''
    if len(dataset) == 0:
        return 0
    count = 0.0
    for i in range(len(dataset)):
        if dataset[i][0] == 0:
            count = count + 1.0
    p=count/len(dataset)
    if p == 0 or p == 1:
        return 0
    else:
      return 2*p*(1-p)


class Node:
    '''
    Helper to construct the tree structure.
    '''

    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1, info={}):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = info
    
    def __str__(self):
        return f'Node {self.label} {self.depth}'


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node(info={'cost': 0, 'data_size': len(data)})
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if not (validation_data is None):
            self._prune_recurs(self.root, validation_data)

    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)

    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)

    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''

        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            # print(prediction == test_Y[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt / len(data)

    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''

        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)

    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        '''
        if node == None:
            return
        else:
            self._prune_recurs(node.left,validation_data)
            self._prune_recurs(node.right,validation_data)
            if node.isleaf == True:
                return
            elif node.right.isleaf == False or node.left.isleaf == False:
                return
            else:
                original_loss = self.loss(validation_data)
                node.isleaf = True
                node.label = 1
                current_loss_1 = self.loss(validation_data)
                node.label = 0
                current_loss_0 = self.loss(validation_data)
                if current_loss_0 > current_loss_1:
                    node.label = 1
                    current_loss = current_loss_1
                else:
                    current_loss = current_loss_0
                    
                if original_loss >= current_loss:
                    node.left = None
                    node.right = None
                    return
                else:
                    node.isleaf = False
                    node.label = -1
        pass

    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the nodex exceede the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf.
            - A label, indicating the label of the leaf (-1 if False)
        '''
        countone = 0
        countzero = 0
        for i in range(len(data)):
            if data[i][0] == 1:
                countone = countone + 1
            else:
                countzero = countzero + 1
        if countone > countzero:
            node.label = 1
        else:
            node.label = 0

        if len(data) == 0:
            node.isleaf = True
            node.label = random.randint(0,1)
            return True, node.label
        elif len(indices) == 0:
            node.isleaf = True
            return True,node.label
        elif len(set([row[0] for row in data])) == 1:
            node.isleaf = True
            return True,node.label
        elif node.depth == self.max_depth :
            node.isleaf = True
            return True,node.label
        else:
            node.label = -1
            return False,-1

    def _split_recurs(self, node, rows, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.
        First use _is_terminal() to check if the node needs to be splitted.
        Then select the column that has the maximum infomation gain to split on.
        Also store the label predicted for this node.
        Then split the data based on whether satisfying the selected column.
        The node should not store data, but the data is recursively passed to the children.
        '''
        a,b = self._is_terminal(node,rows,indices)
        if a == True:
            node.label = b
            return

        max_gain_col = 1
        max_gain = float('-inf')
        for i in indices:
            gain = self._calc_gain(rows, i, self.gain_function)
            if gain > max_gain:
                max_gain = gain
                max_gain_col = i

        node.index_split_on = max_gain_col
        node.isleaf = False
        # node.info = {'cost': 0, 'data_size': len(tl)}

        tl = []
        fl = []
        for n in range(len(rows)):
            if rows[n][max_gain_col] == 1:
                tl.append(rows[n])
            else:
                fl.append(rows[n])
        
        left_node = Node(
            left=None,
            right=None,
            depth=node.depth + 1,
            isleaf=False,
            info={'cost': 0, 'data_size': len(tl)}
        )
        node.left = left_node

        right_node = Node(
            left=None,
            right=None,
            depth=node.depth + 1,
            isleaf=False,
            info={'cost': 0, 'data_size': len(fl)}
        )
        node.right = right_node

        indices.remove(max_gain_col)
        indices2 = copy.deepcopy(indices)

        self._split_recurs(left_node, tl, indices)
        self._split_recurs(right_node, fl, indices2)

    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - (P[x_i=True] * C(P[y=1|x_i=True]) + P[x_i=False]C(P[y=1|x_i=False)])
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''
        tl = []
        fl = []


        for i in range(len(data)):
            if data[i][split_index] == 1:
                tl.append(data[i])
            else:
                fl.append(data[i])

        if len(data) == 0:
            return 0
        else:
            ig = gain_function(data) - (len(fl) / len(data) * gain_function(fl) + len(tl) / len(data) * gain_function(tl))
            return ig

    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        temp = []
        output = []
        print('---START PRINT TREE---')

        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = %d; cost = %f; sample size = %d' % (
                    node.index_split_on, node.info['cost'], node.info['data_size'])
            left = indent + 'T -> ' + print_subtree(node.left, indent + '\t\t')
            right = indent + 'F -> ' + print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')

    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec) / len(data)

    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
