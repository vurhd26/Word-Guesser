import numpy as np
class TreeNode:
    def __init__(self, bigram=None, left=None, right=None, is_leaf=False, prediction=None):
        self.bigram = bigram
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.prediction = prediction

class DecisionTree:
    def __init__(self, max_depth = 15):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, word_list):
        bigram_data = self.prepare_data(word_list)
        used_bigrams = set()
        self.tree = self.build_tree(bigram_data, used_bigrams)
        return self  
    
    def generate_bigrams(self, word):
        return [word[i:i+2] for i in range(len(word) - 1)]
    
    def prepare_data(self, word_list):
        bigram_dict = {}
        for word in word_list:
            bigrams = self.generate_bigrams(word)
            unique_bigrams = sorted(set(bigrams))
            bigram_dict[word] = unique_bigrams[:5]  
        return bigram_dict
    
    def find_most_frequent_bigram(self, bigram_dict, used_bigrams):
        bigram_counts = {}
        for bigrams in bigram_dict.values():
            for bigram in bigrams:
                if bigram not in used_bigrams:
                    if bigram in bigram_counts:
                        bigram_counts[bigram] += 1
                    else:
                        bigram_counts[bigram] = 1
        
        if not bigram_counts:
            return None, 0
        
        most_frequent_bigram = max(bigram_counts, key=bigram_counts.get)
        return most_frequent_bigram, bigram_counts[most_frequent_bigram]
    
    def split_dataset(self, bigram_dict, bigram):
        left_split = {}
        right_split = {}
        for word, bigrams in bigram_dict.items():
            if bigram in bigrams:
                left_split[word] = bigrams
            else:
                right_split[word] = bigrams
        return left_split, right_split
    
    def build_tree(self, bigram_dict, used_bigrams, depth=0):
        if len(bigram_dict) <= 1 or depth >= self.max_depth:
            prediction = list(bigram_dict.keys())
            #print(len(prediction))``
            return TreeNode(is_leaf=True, prediction=prediction)
        
        most_frequent_bigram, freq = self.find_most_frequent_bigram(bigram_dict, used_bigrams)
        #print(most_frequent_bigram, freq)

        while most_frequent_bigram and freq == len(bigram_dict):
            used_bigrams.add(most_frequent_bigram)
            most_frequent_bigram, freq = self.find_most_frequent_bigram(bigram_dict, used_bigrams)
        
        if not most_frequent_bigram or freq == 0:
            prediction = list(bigram_dict.keys())
            return TreeNode(is_leaf=True, prediction=prediction)
        
        used_bigrams.add(most_frequent_bigram)
        
        left_used_bigrams = used_bigrams.copy()
        right_used_bigrams = used_bigrams.copy()
        
        left_split, right_split = self.split_dataset(bigram_dict, most_frequent_bigram)
        
        left_node = self.build_tree(left_split, left_used_bigrams, depth + 1)
        right_node = self.build_tree(right_split, right_used_bigrams, depth + 1)
        
        return TreeNode(bigram=most_frequent_bigram, left=left_node, right=right_node)
    
    def predict(self, tree, bigrams):
        if tree.is_leaf:
            return tree.prediction
        if tree.bigram in bigrams:
            return self.predict(tree.left, bigrams)
        else:
            return self.predict(tree.right, bigrams)

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT PERFORM ANY FILE IO IN YOUR CODE

# DO NOT CHANGE THE NAME OF THE METHOD my_fit or my_predict BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc

################################
# Non Editable Region Starting #
################################
def my_fit( words ):
################################
#  Non Editable Region Ending  #
################################
	dt = DecisionTree(max_depth=100)
	model = dt.fit(words)
	# Do not perform any file IO in your code
	# Use this method to train your model using the word list provided
	
	return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( model, bigram_list ):
################################
#  Non Editable Region Ending  #
################################
	guess_list = model.predict(model.tree, bigram_list)
	# Do not perform any file IO in your code
	# Use this method to predict on a test bigram_list
	# Ensure that you return a list even if making a single guess
	
	return guess_list					# Return guess(es) as a list
