# Author: Kaituo Xu, Fan Yu
import numpy as np


def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # 前向概率矩阵
    alphas = np.zeros((N, T))
    for t in range(T):
        for i in range(N):
            if t == 0:
                alphas[i][t] = pi[i] * B[i][O[t]]
            else:
                alphas[i][t] = np.dot([alpha[t - 1] for alpha in alphas], [a[i] for a in A]) * B[i][O[t]]

    prob = np.sum([alpha[T - 1] for alpha in alphas])

    # End Assignment
    return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # 后向概率矩阵
    betas = np.zeros((N, T))

    for i in range(N):
        betas[i][0] = 1

    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                betas[i][t] += A[i][j]*B[j][O[T-t]]*betas[j][t-1]

    for i in range(N):
        prob += pi[i]*B[i][O[0]]*betas[i][-1]

    # End Assignment
    return prob


def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment
    # 记录每个时刻每个状态的最优路径的概率
    deltas = np.zeros((N, T), dtype=np.float64)
    # 记录每个时刻每个状态前一时刻的概率最大的路径节点
    nodes = np.zeros((N, T), dtype=np.int)

    for i in range(N):
        deltas[0][i] = pi[i] * B[i][0]
    for t in range(1, T):
        for i in range(N):
            tmp = [deltas[t - 1][j] * A[j][i] for j in range(N)]
            nodes[t][i] = int(np.argmax(tmp))
            deltas[t][i] = tmp[nodes[t][i]] * B[i][O[t]]

    best_path = np.zeros((T), dtype=np.int)
    best_path[T - 1] = np.argmax(deltas[T - 1])
    for t in range(T - 2, -1, -1):
        best_path[t] = nodes[t + 1][best_path[t + 1]]

    best_prob = deltas[best_path[1]][best_path[-1]]

    # End Assignment
    return best_prob, best_path


if __name__ == "__main__":
    color2id = {"RED": 0, "WHITE": 1}
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model)
    print(best_prob, best_path)
