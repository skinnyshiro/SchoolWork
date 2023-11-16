import numpy as np

def build_nparray(data):
    samples = np.copy(data[1:,:-1]) #np.empty([data.shape[0]-1,data.shape[1]-1],float)
    samples = samples.astype(float)
    
    labels = np.copy(data[1:,-1])
    labels = labels.astype(int)
    return samples, labels

def build_list(data):
    samples = []
    # I need to iterate by row, so first let's get the number of rows using shape
    for row in range(1,data.shape[0]-1):
        # Now I need to create a sublist to append my main list with 
        new_row = []
        for elem in data[row,:data.shape[1]-1]:
            new_row.append(elem.astype(float))
        samples.append(new_row)
    
    labels = []
    for elem in data[1:,-1]:
        labels.append(elem.astype(int))

    return samples, labels



def build_dict(data):
    samples = {}
    labels = {}
    for row in range(1,data.shape[0]):
        sub_dict = {}
        for column in range(data.shape[1]-1):
            sub_dict[data[0,column]]=data[row,column].astype(float)
        samples[row-1]=sub_dict
        labels[row-1]=data[row,-1]
    
    return samples, labels


