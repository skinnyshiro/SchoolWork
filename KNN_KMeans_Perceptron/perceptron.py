import numpy as np
import matplotlib.pyplot as plt

def perceptron_train(X,Y):
    weights = np.zeros(X.shape[1])
    bias = 0
    epochs = 0
    updatedFlag = True
    while updatedFlag == True and epochs < 100:
        epochs += 1
        #print(f"Epoch #: {epochs}")
        updatedFlag = False
        for sample, values in enumerate(X):
            activation = np.dot(weights,values) + bias
            #print(activation)
            if Y[sample] * activation <= 0:
                updatedFlag = True
                for index, weight in enumerate(weights):
                    weights[index] = weight + Y[sample]*values[index]
                bias = bias + Y[sample]
                #print(f"Updated Weights: {weights}; Updated Bias: {bias}")

    return [weights,bias]


def perceptron_test(X_test,Y_test, w, b):
    prediction_array = np.array([])
    for values in X_test:
        activation = np.dot(w,values) + b
        if activation > 0:
            prediction_array = np.append(prediction_array,1)
        else:
            prediction_array = np.append(prediction_array,-1)
            
    accuracy = np.mean(Y_test == prediction_array)
    return accuracy


#Ask for help on this part, because I'm not sure
def decision_boundary(w,b):
    print(f"W is {w}")
    print(f"b is {b}")
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    step_size = .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))
    xy = np.c_[xx.ravel(), yy.ravel()]
    z = np.dot(xy, w) + b
    z = np.sign(z)
    z = z.reshape(xx.shape)
    
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.6)

    
    # Add labels and a legend
    plt.xlabel('X1')
    plt.ylabel('X2')
    #plt.legend(loc='upper left')

    # Show the plot
    plt.title('Perceptron Decision Boundary')
    plt.grid(True) 
    plt.show()
    return 


##### DELETE LATER #####

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

# X,Y = load_data("perceptron_2.csv")

# W = perceptron_train(X,Y)
# print(W)
# acc = perceptron_test(X,Y,W[0],W[1])
# print("Percept:", acc)

# decision_boundary(W[0],W[1])
