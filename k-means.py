"""
欠点
    Kがわからない
    局所最小
"""
import numpy as np
import matplotlib.pyplot as plt

def d(u, v):
    diff = u - v
    return diff.dot(diff)

def cost(X, R, M):
    cost = 0
    for k in range(len(M)):
        for n in range(len(X)):
            cost += R[n, k] * d(M[k], X[n])
    return cost


def k_means(X, K, max_iter=20, beta=1.0):
    """
    X:data, K:num of cluster
    """

    N, D = X.shape
    M = np.zeros((K, D))
    R = np.zeros((N, K))

    #randomに割り振る
    for k in range(K):
        M[k] = X[np.random.choice(N)] #データを一点だけ取得
    
    costs = np.zeros(max_iter)
    for i in range(max_iter):
        for k in range(K):
            #距離による各サンプルが所属するクラスの割合を計算
            for n in range(N):
                R[n, k] = np.exp(-beta * d(M[k], X[n])) / np.sum(np.exp( -beta*d(M[j], X[n])) for j in range(K))
                
            
            #加重平均点算出
        for k in range(K):
            M[k] = R[:, k].dot(X) / R[:, k].sum()
            
        costs[i] = cost(X, R, M)
        if i > 0:
            if np.abs(costs[i] - costs[i - 1]) < 0.1:
                break
    """
    plt.figure()
    plt.plot(costs)
    plt.title("Costs")
    plt.savefig("./fig/k-means2.png")
    
    plt.figure()
    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.savefig("./fig/k-means3")

    """


def main():
    D = 2
    mu1 = np.array([0, 0])
    mu2 = np.array([5, 5])
    mu3 = np.array([0, 5])

    N = 900
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3

    #plt.scatter(X[:,0], X[:, 1])
    #plt.savefig('./fig/kmean1.png')

    k_means(X, 3, max_iter=20, beta=1.0)
if __name__ == "__main__":
    main()