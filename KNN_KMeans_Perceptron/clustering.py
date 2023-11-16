import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

def K_Means(X,K,mu=None):
    if X.ndim == 1:
        X = X.reshape(-1,1)
    while mu is None or len(mu) < K:
        random_sample_index = np.random.choice(range(len(X)),1)
        #print(random_sample_indices)
        cluster_center = X[random_sample_index]
        if mu is None:
            mu = np.array(cluster_center)
        elif not np.any(np.all(cluster_center == mu, axis=1)):
            mu = np.vstack((mu, cluster_center))
    #print(f"Mu: {mu}")
    
    if len(mu) != K:
        print("Invalid Input: K != Len(Mu) ")
        return
    converged = False
    epoch = 0
    while converged == False and epoch <20:
        epoch += 1
        k_cluster_dict = {cluster: [] for cluster in range(K)}
        new_mu = np.zeros(mu.shape)
        sample_distances = spatial.distance.cdist(X,mu)
        #print(sample_distances)
        for sample, distances in enumerate(sample_distances):
           k_cluster_dict[np.argmin(distances)].append(X[sample])
        #print(k_cluster_dict)
        for cluster, samples in k_cluster_dict.items():
            #print(f"Cluster {cluster}'s New Mean: {np.mean(samples,axis=0)}")
            #print(f"Cluster {cluster}: {samples}")
            new_mu[cluster]=np.mean(samples,axis=0)
        #print(f"New Mu: {new_mu}")
        if np.array_equal(mu,new_mu):
            #print("Clusters have converged.")
            converged = True
        else:
            mu = new_mu
    return mu

def K_Means_better(X,K):
    unique_mu_dict = {}
    count = 1000
    while count > 0:
        count -= 1
        mu = np.sort(K_Means(X,K),axis=0)
        tuple_mu = tuple(tuple(row) for row in mu)
        if tuple_mu in unique_mu_dict:
            unique_mu_dict[tuple_mu]+=1
        else:
            unique_mu_dict[tuple_mu]=1
    #print(unique_mu_dict)
    best_mu = max(unique_mu_dict, key=unique_mu_dict.get)
    #print(best_mu)
    best_mu_array = np.array(best_mu)
    return best_mu_array


#### DELETE LATER ####


# X = np.genfromtxt("clustering_2.csv", skip_header=1, delimiter=',')

#mu = np.array([[1],[5]])
# mu = K_Means(X,3)
# print("KMeans:", mu)

#print(f"K Means Best Cluster: {K_Means_better(X,3)}")

# k_2_clusters = K_Means_better(X,2)
# k_3_clusters = K_Means_better(X,3)

# plt.scatter(X[:,0],X[:,1],marker=".",color='black',label="Data")
# plt.scatter(k_2_clusters[:,0],k_2_clusters[:,1],marker='x',color='green',label="K=2")
# plt.scatter(k_3_clusters[:,0],k_3_clusters[:,1],marker='P',color='red',label="K=3")

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()

# plt.title('K Means Clusters by K')
# plt.grid(True)
# plt.show()



