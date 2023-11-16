from sklearn import tree
import numpy as np


def adaboost_train(X, Y, max_iter):
    f = np.array([])
    alpha_array = np.full(len(Y),1/len(Y))
    #print(f'Alpha Array: {alpha_array}')
    num_per_sample = np.ones_like(alpha_array)
    x_to_train = X #We separate these variables from the initial ones passed so that we can change them over iterations
    y_to_train = Y
    
    for iter in range(max_iter):
        clf = tree.DecisionTreeClassifier(max_depth=1)
        clf.fit(x_to_train,y_to_train)
        f = np.append(f,clf)
        pred = clf.predict(X,Y)
        #print(f'Predictions: {pred}')
        error_list = np.array(Y) != np.array(pred)
        error_list = error_list.astype(int)
        #print(f'Error List: {error_list}')
        correct_samples = []
        incorrect_samples = []
        for index, value in enumerate(error_list):
            if value == 0:
                correct_samples.append(index)
            else:
                incorrect_samples.append(index)
        #print(f'Correct Samples: {correct_samples}\nIncorrect Samples: {incorrect_samples}')
        alpha_array[correct_samples] = alpha_array[correct_samples]/(np.sum(alpha_array[correct_samples])*2)
        alpha_array[incorrect_samples] = alpha_array[incorrect_samples]/(np.sum(alpha_array[incorrect_samples])*2)
        #print(f'New Weights: {alpha_array}')
        num_per_sample = (alpha_array/min(alpha_array)).astype(int)
        #print(f'Number per Sample: {num_per_sample}')
        #print(f'Alpha Array: {alpha_array}')
        # Creates the new x & y based on the numbers per sample from the weights
        x_to_train = [value for value, count in zip(X,num_per_sample) for _ in range(count)]
        y_to_train = [value for value, count in zip(Y,num_per_sample) for _ in range(count)]
        #print(f'New X values: {x_to_train}\nNew Y values: {y_to_train}')
        

    return f, alpha_array





def adaboost_test(X, Y, f, alpha):
    #print(f'Array of Stumps: {f}')
    predictions_array = []
    for clf in f: 
        pred = clf.predict(X,Y)
        predictions_array.append(pred)
    
    #print(predictions_array)
    voted_pred = []
    for sample in range(len(Y)):
        votes = []
        for preds in predictions_array:
            votes.append(preds[sample])
        #print(f'Votes: {votes}, Prediction: {np.sign(sum(votes))}')
        voted_pred.append(np.sign(sum(votes)))
    #print(f'Voted Predictions: {voted_pred}')
    error_list = np.array(Y) != np.array(voted_pred)
    error_list = error_list.astype(int)
    total_error = sum(error_list*alpha)
    #print(f'Error List: {error_list}\nAlpha: {alpha}\nTotal Error: {total_error}')
    acc = 1 - total_error # 1 - (sum(error_list) / len(error_list))

    return acc



# X = [[-2,-2],[-3,-2],[-2,-3],[-1,-1],[-1,0],[0,-1],[1,1],[1,0],[0,1],[2,2],[3,2],[2,3]]
# Y=[-1,-1,-1,1,1,1,-1,-1,-1,1,1,1]
# f, alpha = adaboost_train(X,Y,3)
# acc = adaboost_test(X,Y,f,alpha)
# print("Accuracy:", acc)
