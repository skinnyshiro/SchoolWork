import numpy as np
from scipy import spatial

def KNN_test(X_train,Y_train,X_test,Y_test,K):
    prediction_list = np.array([])
    for test_sample, distance_list in enumerate(spatial.distance.cdist(X_test,X_train)):
        indexed_distance_list = [(neighbor, distance) for neighbor, distance in enumerate(distance_list)]
        sorted_distances = sorted(indexed_distance_list, key=lambda x: x[1])
        #print(f"Sample {sample}: {sorted_distances}")
        k_nearest_list = [neighbor[0] for neighbor in sorted_distances[:K]]
        #print(f"Sample {sample}: KNN {k_nearest_list}")
        k_nearest_labels = [Y_train[neighbor] for neighbor in k_nearest_list]
        if sum(k_nearest_labels) >= 0:
            prediction_list = np.append(prediction_list,1)
        else:
            prediction_list = np.append(prediction_list,-1)
    #print(f"Labels: {Y_test}")
    #print(f"Predictions: {prediction_list}")
    accuracy = np.mean(Y_test == prediction_list)
    return accuracy


def choose_K(X_train,Y_train,X_val,Y_val):
    best_acc = 0
    best_K = None
    for K in range(3,len(Y_train),2):
        #print(K)
        acc = KNN_test(X_train,Y_train,X_val,Y_val,K)
        if acc > best_acc:
            best_acc = acc
            best_K = K
            #print(f"K: {K}, New Best Acc: {best_acc}")
    
    return best_K








#### DELETE LATER ####
# Note to self, currently the Train and Test data are the same, so the first nearest neighbor will always be itself. Awaiting response from Sara


# def load_data(file_data):
#     data = np.genfromtxt(file_data, skip_header=1, delimiter=',')
#     X = []
#     Y = []
#     for row in data:
#         temp = [float(x) for x in row]
#         temp.pop(-1)
#         X.append(temp)
#         Y.append(int(row[-1]))
#     X = np.array(X)
#     Y = np.array(Y)
#     return X,Y

# X,Y = load_data("nearest_neighbors_1.csv")
# X_2, Y_2 = load_data("nearest_neighbors_2.csv")
# X_3, Y_3 = load_data("nearest_neighbors_3.csv")

#acc = KNN_test(X,Y,X,Y,5)
#acc = KNN_test(X_3,Y_3,X_3,Y_3,3)
#print("KNN:", acc)

# distances = spatial.distance.cdist(X_2,X)
# for index, distance_list in enumerate(distances):
#     print(f"Sample {index}: {distance_list}")
    
#print("Best K: ",choose_K(X,Y,X,Y))