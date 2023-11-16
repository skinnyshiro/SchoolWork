import math
import numpy as np
import random

class Node:
    """
    Capture data about each node in a tree
    """
    def __init__(
            self,
            X: np.array,
            Y: np.array,
            max_depth = None,
            mode = None,
            depth = None,
            split_rule = None,
            node_direction = None
    ):
        # Save data to node
        self.X = X
        self.Y = Y

        # Save hyperparameter
        self.max_depth = max_depth

        # Save current depth of node or set to 0 if root
        self.depth = depth if depth else 0

        # Set the mode of the tree to either binary or real
        self.mode = mode

        # Assign the features
        try:
            self.features = self.X.shape[1] 
        except IndexError:
            self.features = len(self.X)

        # Establish left & right nodes, as well as node direction, or if starting node "root"
        self.left = None
        self.right = None
        self.node_direction = node_direction if node_direction else "root"

        # Establish how a node will be split
        self.split_rule = split_rule if split_rule else ""

        # Count number of samples
        self.num_samples = len(Y)

        # Get the entropy of a node
        self.entropy = self.calc_entropy(X,Y)

        # Select the dominant label of the node
        self.dominant_label = 1 if (self.Y.sum()/self.num_samples) >= 0.5 else 0 # Assigns the dominant label for the leaf based on percentage. If 50/50, assigns positive label

        # Set the best feature for a split
        self.split_feature = None

    
    def calc_entropy(self,X,Y):
        n = len(Y)
        zero_count = n-Y.sum() # Subtract the sum of 1-labels from the total
        one_count = Y.sum() # Sum of 1 labels
        #print(zero_count) #TODO: Delete
        #print(one_count) #TODO: Delete
        prob_zero = zero_count/n
        prob_one = one_count/n
        entropy_zero = 0 if prob_zero == 0 else -1*prob_zero*math.log2(prob_zero) #Calculates entropy for 0-label and resolves negative infinity value error
        entropy_one = 0 if prob_one == 0 else -1*prob_one*math.log2(prob_one) #Calculates entropy for 1-label and resolves negative infinity value error
        entropy_total = entropy_zero + entropy_one 
        #print(entropy_total) #TODO: Delete
        return entropy_total
    
    def find_split(self):
        parent_entropy = self.entropy
        
        best_ig = 0
        best_feature = None
        best_left_samples = None
        best_left_labels = None
        best_right_samples = None
        best_right_labels = None
        
        if self.mode == "binary":
            for feature in range(self.features):
                left_samples = np.array([])
                left_labels = np.array([])
                right_samples = np.array([])
                right_labels = np.array([])
                for index,sample in enumerate(self.X):
                    #print(index)
                    #print(sample)
                    #print(sample[feature])
                    #print(self.Y[index])
                    if sample[feature] == 0:
                        if left_samples.size == 0:
                            left_samples = np.append(left_samples,sample)
                        else:
                            left_samples = np.vstack((left_samples,sample))
                        left_labels = np.append(left_labels,self.Y[index])
                    else:
                        if right_samples.size == 0:
                            right_samples = np.append(right_samples,sample)
                        else:
                            right_samples = np.vstack((right_samples,sample))
                        right_labels = np.append(right_labels,self.Y[index])
                #print(f"Left Labels: {left_labels}")
                #print(f"Right Labels: {right_labels}")
                #print(f"Left Samples: {left_samples}")
                #print(f"Right Samples: {right_samples}")
                #print(f"Left Entropy: {self.calc_entropy(left_samples,left_labels)}")
                left_entropy = 0 if len(left_labels) == 0 else len(left_labels) / self.num_samples * self.calc_entropy(left_samples,left_labels)
                right_entropy = 0 if len(right_labels) == 0 else len(right_labels) / self.num_samples * self.calc_entropy(right_samples,right_labels)
                split_entropy = left_entropy + right_entropy
                info_gain = parent_entropy - (split_entropy)
                #print(f"Info Gain for Split on Feature {feature}: {info_gain}")
                if info_gain > best_ig:
                    best_ig = info_gain
                    best_feature = feature
                    best_left_samples = left_samples
                    best_left_labels = left_labels
                    best_right_samples = right_samples
                    best_right_labels = right_labels

        elif self.mode == 'real':
            feature_means = np.array([])
            for feature in range(self.features):
                feature_means = np.append(feature_means,np.mean(self.X[:,feature]))
            #print(f"List of Feature Means: {feature_means}")
            for feature, feature_mean in enumerate(feature_means):
                left_samples = np.array([])
                left_labels = np.array([])
                right_samples = np.array([])
                right_labels = np.array([])
                for index,sample in enumerate(self.X):
                    if sample[feature] < feature_mean:
                        if left_samples.size == 0:
                            left_samples = np.append(left_samples,sample)
                        else:
                            left_samples = np.vstack((left_samples,sample))
                        left_labels = np.append(left_labels,self.Y[index])
                    else:
                        if right_samples.size == 0:
                            right_samples = np.append(right_samples,sample)
                        else:
                            right_samples = np.vstack((right_samples,sample))
                        right_labels = np.append(right_labels,self.Y[index])
                left_entropy = 0 if len(left_labels) == 0 else len(left_labels) / self.num_samples * self.calc_entropy(left_samples,left_labels)
                right_entropy = 0 if len(right_labels) == 0 else len(right_labels) / self.num_samples * self.calc_entropy(right_samples,right_labels)
                split_entropy = left_entropy + right_entropy
                info_gain = parent_entropy - (split_entropy)
                #print(f"Info Gain for Split on Feature {feature}: {info_gain}")
                if info_gain > best_ig:
                    best_ig = info_gain
                    best_feature = feature
                    #print(f"Feature Mean for Split: {feature_mean}")
                    best_left_samples = left_samples
                    best_left_labels = left_labels
                    best_right_samples = right_samples
                    best_right_labels = right_labels


        return best_feature, best_left_samples, best_left_labels, best_right_samples, best_right_labels #TODO: may need to modify later for Real values
    
    def build_tree(self):
        
        #print(f"Depth: {self.depth}")
        #print(f"Node Direction: {self.node_direction}")
        #print(f"Number of Samples: {self.num_samples}")
        
        if self.depth != max_depth and self.num_samples > 1: # Using != instead of < because we need to account for -1, stops splitting when only a single sample or less remains

            best_feature, best_left_samples, best_left_labels, best_right_samples, best_right_labels = self.find_split()
            #print(f"Best Left Samples: {best_left_samples}")
            if best_feature is not None:
                self.split_feature = best_feature

                # Create Left Node
                left_node = Node(
                    best_left_samples, 
                    best_left_labels, 
                    self.max_depth,
                    self.mode,
                    depth = self.depth + 1,
                    node_direction = "left_node",
                    split_rule = f"{best_feature} == 0" if self.mode == "binary" else f"{best_feature} < {np.mean(self.X[:,best_feature])}"
                )
                
                
                self.left = left_node
                self.left.build_tree()

                # Create Right Node
                right_node = Node(
                    best_right_samples, 
                    best_right_labels, 
                    self.max_depth,
                    self.mode,
                    depth = self.depth + 1,
                    node_direction = "right_node",
                    split_rule = f"{best_feature} == 1" if self.mode == "binary" else f"{best_feature} >= {np.mean(self.X[:,best_feature])}"
                    )
                
                self.right = right_node
                self.right.build_tree()


        



def DT_train_binary(X,Y,max_depth):
    DT = Node(X,Y,max_depth,"binary")
    DT.build_tree()
    return DT

def DT_test_binary(X,Y,DT):
    y_pred = np.zeros_like(Y) # Set array equal to Y's size filled with zeros. Ones will represent matching labels
    for index, sample in enumerate(X):
        prediction = DT_make_prediction(sample,DT)
        if prediction == Y[index]:
            y_pred[index] = 1
    #print(y_pred)
    accuracy = sum(y_pred)/len(Y)
    return accuracy

def DT_make_prediction(x,DT):
    current_node = DT
    if current_node.mode == 'binary':
        while current_node.depth != current_node.max_depth:
            if current_node.left is None and current_node.right is None:
                break
            
            elif x[current_node.split_feature] == 0:
                current_node = current_node.left

            else:
                current_node = current_node.right
    
    elif current_node.mode == 'real':
        while current_node.depth != current_node.max_depth:
            if current_node.left is None and current_node.right is None:
                break
            
            elif x[current_node.split_feature] < np.mean(current_node.X[:,current_node.split_feature]):
                current_node = current_node.left

            else:
                current_node = current_node.right

    return current_node.dominant_label

def DT_train_real(X,Y,max_depth):
    DT = Node(X,Y,max_depth,"real")
    DT.build_tree()
    return DT

def DT_test_real(X,Y,DT):
    y_pred = np.zeros_like(Y) # Set array equal to Y's size filled with zeros. Ones will represent matching labels
    for index, sample in enumerate(X):
        prediction = DT_make_prediction(sample,DT)
        if prediction == Y[index]:
            y_pred[index] = 1
    #print(y_pred)
    accuracy = sum(y_pred)/len(Y)
    return accuracy

def RF_build_random_forest(X,Y,max_depth,num_of_trees):
    n_rows = len(Y)
    n_sample = int(0.1*n_rows)
    forest = np.array([])
    for tree in range(num_of_trees):
        random_indices = np.random.permutation(n_rows)
        sampled_indices = random_indices[:n_sample]
        validation_X = X[sampled_indices]
        #print(validation_X)
        validation_Y = Y[sampled_indices]
        DT = DT_train_binary(validation_X,validation_Y,max_depth)
        print(f"DT {tree}: {DT_test_binary(validation_X,validation_Y,DT):0.5f}")
        forest = np.append(forest,DT)
    return forest

def RF_test_random_forest(X,Y,RF):
    y_pred_match = np.zeros_like(Y)
    for index,sample in enumerate(X):
        tree_votes = np.array([])
        for tree in RF:
            tree_votes = np.append(tree_votes,DT_make_prediction(sample,tree))
        #print(f"Tree Votes: {tree_votes}")
        if sum(tree_votes)/len(tree_votes) >= .5:
            if Y[index] == 1:
                y_pred_match[index] = 1
        else:
            if Y[index] == 0:
                y_pred_match[index] = 1
    #print(y_pred_match)
    accuracy = sum(y_pred_match)/len(y_pred_match)
    return accuracy



##### Testing

import data_storage as ds
#file_name = "cat_dog_data.csv"
#file_name = "data_2.csv"
file_name = "haberman.csv"


data = np.genfromtxt(file_name, dtype=str, delimiter=',')

a_samples,a_labels = ds.build_nparray(data)
#print(type(a_labels))
#print(a_samples)
#print(a_labels,"\n")

#a_labels = np.array([1,1,1,1,1,1])

max_depth = 1

#DT = DT_train_binary(a_samples,a_labels,max_depth)
#DT = DT_train_real(a_samples,a_labels,max_depth)

#print(f"Entropy: {DT.entropy}")
#print(f"Dominant Label: {DT.dominant_label}")
#print(f"Best Feature for Split: {DT.find_split()}")

#DT.build_tree()

#test = np.array([0,0,1,1,0])
#print(test.shape[1])

#print(f"Prediction for Test: {DT_make_prediction(test,DT)}")

#print(f"Accuracy: {DT_test_binary(a_samples,a_labels,DT)}")
#print(f"Accuracy: {DT_test_real(a_samples,a_labels,DT)}")

# Note: How to iterate over columns
#for feature in range(a_samples.shape[1]):
#    print(np.mean(a_samples[:,[feature]]))

#print(np.mean(test))

# num_of_trees = 11
# RF = RF_build_random_forest(a_samples,a_labels,max_depth,num_of_trees)
# RF_accuracy = RF_test_random_forest(a_samples,a_labels,RF)
# print(f"RF: {RF_accuracy}")