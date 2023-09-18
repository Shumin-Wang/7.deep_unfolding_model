import numpy as np
# import torch
import generateConstrains

def generateState():
    theta_t = [30]
    # theta_r1 = np.linspace(10, 30, 3)
    # theta_r2 = np.linspace(40, 60, 3)
    theta_r1 = [10]
    theta_r2 = [40]
    # theta_r1 = np.linspace(10, 12, 5)
    # theta_r2 = np.linspace(40, 42, 5)
    x_train = np.zeros([len(theta_t) * len(theta_r1) * len(theta_r2), 3])
    for tt in range(len(theta_t)):
        for r1 in range(len(theta_r1)):
            for r2 in range(len(theta_r2)):
                x_train[tt * len(theta_r1) * len(theta_r1) + (r1) * len(theta_r2) + r2,:] = ([theta_t[tt], theta_r1[r1], theta_r2[r2]])

    return x_train

states = generateState()

def generateChannel(state, dist_ris,D,Nt,Nr,Nris,K,f):

    lt = dist_ris / np.tan(state[0] / 180 * np.pi)
    lr1 = (D - dist_ris) / np.cos(state[1] / 180 * np.pi)
    lr2 = (D - dist_ris) / np.cos(state[2] / 180 * np.pi)

    D2 = np.sqrt(lr1 ** 2 + (D - dist_ris) ** 2)

    [Hdir1, U, G1] = generateConstrains.chan_mat_RIS_new_model_ob(Nt, Nr, Nris, lt, lr1, D, 1,
                                                                  K, f, dist_ris)
    [Hdir2, U, G2] = generateConstrains.chan_mat_RIS_new_model_ob(Nt, Nr, Nris, lt, lr2, D, 1,
                                                                  K, f, dist_ris)

    return U, G1, G2

def generateAllChannel(states, dist_ris = 20, D = 50, Nt = 8, Nr = 2, Nris = 100, K = 100000, f = 27e9):

    U = np.zeros([len(states), Nris, Nt], dtype=np.complex128)
    G1 = np.zeros([len(states),Nr, Nris], dtype=np.complex128)
    G2 = np.zeros([len(states),Nr, Nris], dtype=np.complex128)

    for i in range(len(states)):
         [U[i,:,:], G1[i,:,:], G2[i,:,:]] = generateChannel(states[i], dist_ris,D,Nt,Nr,Nris,K,f)

    return U, G1, G2



