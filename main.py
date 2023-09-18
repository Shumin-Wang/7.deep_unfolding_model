import time

import torch.nn as nn
import torch
import scipy.io as scio
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import math
import mat73
import random
import datetime
import generateConstrains
import generalState
import MyFadingChannel



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')
print(torch.version.cuda)
print(torch.cuda.is_available())



dd = 0.5
N_t = 8
N_r = 2
N_ris = 10*10
# lt = 0000
D = 50
K = 100000
f = 27e9
dist_ris = 20

threshold = -120
noise = 10**(-120/10)

D2 = D - dist_ris



class loss_function(nn.Module):

    def __init__(self):
        super(loss_function, self).__init__()

    def forward(self,output,target):

        n_sample = len(output)
        loss = 0
        for i in range(n_sample):
            loss = loss+(torch.norm(output[i,:,:]-target[i,:,:])**2) #/torch.norm(target[i,:,:])**2
        return loss/n_sample



# generate the training data
def generate_data():

    x_train = generalState.generateState()

    x_train = torch.from_numpy(x_train)


    return x_train

# define the iterator
# def data_iter(batch_size, x_train, y_train, N_sample_trainC):
#     indices = list(range(N_sample_train))
#     for i in range(0, N_sample_train, batch_size):
#         j = torch.LongTensor(indices[i: min(i + batch_size, N_sample_train)])  # 最后一次可能不足一个batch
#         yield x_train.index_select(0, j), y_train.index_select(0, j)

def preprocessing_data():

    # one-hot encoding from -90 to 90, interval equals 1
    x_t = torch.arange(10, 60.5, 0.5)
    x_r1 = torch.arange(10, 60.5, 0.5)
    x_r2 = torch.arange(10,60.5,0.5)

    N_sample_train = len(x_t) * len(x_r1) * len(x_r2)
    x = torch.zeros(N_sample_train, 3)
    # x2 = torch.zeros(N_sample_train, 2)
    for t in range(len(x_t)):
        for r in range(len(x_r1)):
            for r2 in range(len(x_r2)):
                x[t*len(x_r1)*len(x_r2)+r * len(x_r2) + r2, :] = torch.tensor([x_t[t], x_r1[r],x_r2[r2]])
            # x2[t * len(x_r1) + r, :] = torch.tensor([x_t[t], x_r2[r]])
    enc = preprocessing.OneHotEncoder()
    enc.fit(x)

    # separate the integer and fraction
    x_train = generate_data()

    x_train_int, x_train_frac = separate_frac_int(x_train)

    x_train_int_encoding = enc.transform(x_train_int)

    # x_train_int_encoding = torch.from_numpy(x_train_int_encoding.A)
    # x_train_frac = np.random.rand(7225,2)

    # x_train_int_encoding1 = processing_train(x_train_int_encoding1, x_train_frac1)
    # x_train_int_encoding1 = torch.from_numpy(x_train_int_encoding1.A)
    #
    # x_train_int2, x_train_frac2 = separate_frac_int(x_train2)
    #
    # x_train_int_encoding2 = enc.transform(x_train_int2)

    # x_train_int_encoding = torch.from_numpy(x_train_int_encoding.A)
    # x_train_frac = np.random.rand(7225,2)

    x_train_int_encoding = processing_train(x_train_int_encoding, x_train_frac)
    x_train_int_encoding = torch.from_numpy(x_train_int_encoding.A)

    return enc, x_train_int_encoding, x_train

def separate_frac_int(x_train):
    x_reshape = x_train.reshape(len(x_train)*3, 1).detach().numpy()

    x_int = np.zeros(len(x_reshape))
    x_frac = np.zeros(len(x_reshape))

    for i in range(len(x_reshape)):
        x_i_frac , x_i = math.modf(x_reshape[i, 0])
        if x_i_frac == 0.5:
          x_int[i] = int(x_i)+0.5
          x_frac[i] = 0
        elif x_i_frac >0.5:
            x_int[i] = int(x_i)+0.5
            x_frac[i] = x_i_frac - 0.5
        else:
            x_int[i] = int(x_i)
            x_frac[i] = x_i_frac

    x_train_int = x_int.reshape(len(x_train), 3)
    x_train_frac = x_frac.reshape(len(x_train), 3)

    return x_train_int, x_train_frac

def processing_train(x_train_int_encdoing, x_train_frac):
    x_index = np.argwhere(x_train_int_encdoing == 1)
    x_train_frac_reshape = x_train_frac.reshape(len(x_train_frac)*3,1)
    for i in range(len(x_index)):
        x_train_int_encdoing[x_index[i,0],x_index[i,1]] = x_train_int_encdoing[x_index[i,0],x_index[i,1]]+x_train_frac_reshape[i]

    return x_train_int_encdoing


enc, x_train, x_train_original= preprocessing_data()

N_sample_train = len(x_train)



# class DNN_Model(torch.nn.Module):
#     def __init__(self,N_ris, N_r, N_t, D, K, f, dist_ris, threshold):
#         super(DNN_Model, self).__init__()
#
#         self.N_ris = N_ris
#         self.N_t = N_t
#         self.N_r = N_r
#         self.D = D
#         self.K = K
#         self.f = f
#         self.dist_ris = dist_ris
#         self.threshold = threshold
#
#
#         self.f1 = torch.nn.Sequential(
#             torch.nn.Linear(303, 1024),
#             torch.nn.ReLU(),
#             torch.nn.Linear(1024, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, N_ris+4*N_r*N_t),
#         )
#
#
#
#
#     def forward(self, sample1, sample2, U):
#         theta1 = self.f1(sample1)
#         M1 = self.N_t
#         N1 = self.N_r
#
#         F1_test_real = theta1[:, 200:200 + M1 * N1]
#         F1_test_imag = theta1[:, 200 + M1 * N1:200 + 2 * M1 * N1]
#
#
#
#         F2_test_real = theta1[:, 200 + 2 * M1 * N1:200 + 3 * M1 * N1]
#         F2_test_imag = theta1[:, 200 + 3 * M1 * N1:]
#
#         F1 = F1_test_real + 1j*F1_test_imag
#         F2 = F2_test_real + 1J*F2_test_imag
#
#         F1 = F1[:,:,None]
#         F2 = F2[:,:,None]
#
#         F1 = F1 * torch.sqrt(2 / (torch.transpose(torch.conj(F1), 1, 2) @ F1))
#         F2 = F2 * torch.sqrt(2 / (torch.transpose(torch.conj(F2), 1, 2) @ F2))
#
#
#
#         theta1[:, 200:200 + M1 * N1] = torch.real(F1.squeeze())
#         theta1[:, 200 + M1 * N1:200 + 2 * M1 * N1] = torch.imag(F1.squeeze())
#
#         theta1[:, 200 + 2 * M1 * N1:200 + 3 * M1 * N1] = torch.real(F2.squeeze())
#         theta1[:, 200 + 3 * M1 * N1:] = torch.imag(F2.squeeze())
#
#         theta_real = theta1[:, 0:100]
#         theta_imag = theta1[:, 100:200]
#
#         theta = theta_real + 1j* theta_imag
#
#         theta = theta/abs(theta)
#
#         # theta = self.ProjectRIS(F1,F2,theta,sample2,U)
#
#         theta1[:, 0:100] = torch.real(theta)
#         theta1[:, 100:200] = torch.imag(theta)
#
#         return theta1
#
#     def ProjectRIS(self, F1, F2, theta, sample2,U):
#         theta_hat = torch.zeros([theta.shape[0], theta.shape[1]],dtype=torch.complex128)
#         for i in range(theta.shape[0]):
#
#             threshold_w = (10 ** ((self.threshold) / 10)) / 1000
#
#             lt = self.dist_ris / torch.sin(torch.pi*sample2[i,0]/180)
#             D2 = self.D - self.dist_ris
#             T_list = generateConstrains.generate_constrains(sample2[i,1], sample2[i,2], torch.reshape(F1[i,:,:],[self.N_t,self.N_r]), torch.reshape(F2[i,:,:],[self.N_t,self.N_r]), 0.5, D2, self.N_t, self.N_r, int(self.N_ris/2), lt, self.D, self.K, self.f, self.dist_ris,U[i,:,:])
#             # G_list_All, theta_all, U = generateConstrains.generate_constrainsChannleAll(0.5, D2, self.N_t, self.N_r, int(self.N_ris/2), lt, self.D, self.K, self.f, self.dist_ris)
#             # T_list = generateConstrains.generate_constrains(sample2[i,1], sample2[i,2], torch.reshape(F1[i,:,:],[self.N_t,self.N_r]), torch.reshape(F2[i,:,:],[self.N_t,self.N_r]), U, G_list_All, theta_all,2)
#             CCC = torch.conj(theta[i,:,None]).T @ (T_list / threshold_w) @ theta[i,:,None]
#             CCC = torch.real(np.squeeze(CCC))
#             if torch.max(CCC) >1:
#                 theta_hat[i,:] = theta[i,:]*torch.sqrt(1/torch.max(CCC))
#             else:
#                 theta_hat[i, :] = theta[i, :]
#         return theta_hat



class DNN_Model(torch.nn.Module):
    def __init__(self,N_ris, N_r, N_t, D, K, f, dist_ris, threshold):
        super(DNN_Model, self).__init__()

        self.N_ris = N_ris
        self.N_t = N_t
        self.N_r = N_r
        self.D = D
        self.K = K
        self.f = f
        self.dist_ris = dist_ris
        self.threshold = threshold


        self.f1 = torch.nn.Sequential(
            torch.nn.Linear(303, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, N_ris+4*N_t*N_r),
        )




    def forward(self, sample1, sample2, U):
        theta1 = self.f1(sample1)
        M1 = self.N_t
        N1 = self.N_r

        F1_test_real = theta1[:, 200:200 + M1 * N1]
        F1_test_imag = theta1[:, 200 + M1 * N1:200 + 2 * M1 * N1]



        F2_test_real = theta1[:, 200 + 2 * M1 * N1:200 + 3 * M1 * N1]
        F2_test_imag = theta1[:, 200 + 3 * M1 * N1:]

        F1 = F1_test_real + 1j*F1_test_imag
        F2 = F2_test_real + 1J*F2_test_imag

        F1 = F1[:,:,None]
        F2 = F2[:,:,None]

        F1 = F1 * torch.sqrt(2 / (torch.transpose(torch.conj(F1), 1, 2) @ F1))
        F2 = F2 * torch.sqrt(2 / (torch.transpose(torch.conj(F2), 1, 2) @ F2))



        theta1[:, 200:200 + M1 * N1] = torch.real(F1.squeeze())
        theta1[:, 200 + M1 * N1:200 + 2 * M1 * N1] = torch.imag(F1.squeeze())

        theta1[:, 200 + 2 * M1 * N1:200 + 3 * M1 * N1] = torch.real(F2.squeeze())
        theta1[:, 200 + 3 * M1 * N1:] = torch.imag(F2.squeeze())

        theta_real = theta1[:, 0:100]
        theta_imag = theta1[:, 100:200]

        theta = theta_real + 1j* theta_imag

        theta = theta/abs(theta)

        theta = self.ProjectRIS(F1,F2,theta,sample2,U)

        theta1[:, 0:100] = torch.real(theta)
        theta1[:, 100:200] = torch.imag(theta)

        return theta1

    def ProjectRIS(self, F1, F2, theta, sample2,U):
        theta_hat = torch.zeros([theta.shape[0], theta.shape[1]],dtype=torch.complex128)
        for i in range(theta.shape[0]):

            threshold_w = (10 ** ((self.threshold) / 10)) / 1000

            lt = self.dist_ris / torch.sin(torch.pi*sample2[i,0]/180)
            D2 = self.D - self.dist_ris
            T_list = generateConstrains.generate_constrains(sample2[i,1], sample2[i,2], torch.reshape(F1[i,:,:],[self.N_t,self.N_r]), torch.reshape(F2[i,:,:],[self.N_t,self.N_r]), 0.5, D2, self.N_t, self.N_r, int(self.N_ris/2), lt, self.D, self.K, self.f, self.dist_ris,U[i,:,:])
            # G_list_All, theta_all, U = generateConstrains.generate_constrainsChannleAll(0.5, D2, self.N_t, self.N_r, int(self.N_ris/2), lt, self.D, self.K, self.f, self.dist_ris)
            # T_list = generateConstrains.generate_constrains(sample2[i,1], sample2[i,2], torch.reshape(F1[i,:,:],[self.N_t,self.N_r]), torch.reshape(F2[i,:,:],[self.N_t,self.N_r]), U, G_list_All, theta_all,2)
            CCC = torch.conj(theta[i,:,None]).T @ (T_list / threshold_w) @ theta[i,:,None]
            CCC = torch.real(np.squeeze(CCC))
            if torch.max(CCC) >1:
                theta_hat[i,:] = theta[i,:]*torch.sqrt(1/torch.max(CCC))
            else:
                theta_hat[i, :] = theta[i, :]
        return theta_hat


# class MyLossFunction(torch.nn.Module):
#     def __init__(self, noise, N_t, N_r):
#         super(MyLossFunction, self).__init__()
#         self.noise = noise
#         self.N_t = N_t
#         self.N_r = N_r
#
#
#
#     def updatePhaseRIS(self,W1, W2, Sigma1, Sigma2,F1,F2, theta, Hr_rx1, Hr_rx2, Hr_tx):
#
#         A1MidValue = W1 @ Hr_rx1
#         A11 = torch.conj(A1MidValue).T @ torch.linalg.inv(Sigma1) @ A1MidValue
#         B11MidValue = Hr_tx @ F1
#         B11 = (B11MidValue @ torch.conj(B11MidValue).T).T
#         A12 = torch.conj(A1MidValue).T @ torch.linalg.inv(Sigma1) @ A1MidValue
#         B12MidValue = Hr_tx @ F2
#         B12 = (B12MidValue @ torch.conj(B12MidValue).T).T
#         b1 = torch.diag(torch.conj(Hr_tx @ F1 @ torch.linalg.inv(Sigma1) @ W1 @ Hr_rx1).T)
#         b2 = torch.diag(torch.conj(Hr_tx @ F2 @ torch.linalg.inv(Sigma2) @ W2 @ Hr_rx2).T)
#         T1 = ((A11 * B11 + A12 * B12))
#
#         A2MidValue = W2 @ Hr_rx2
#         A21 = torch.conj(A2MidValue).T @ torch.linalg.inv(Sigma2) @ A2MidValue
#
#         B21 = (B12MidValue @ torch.conj(B12MidValue).T).T
#
#
#         A22 = torch.conj(A2MidValue).T @ torch.linalg.inv(Sigma2) @ A2MidValue
#         B22 = (B11MidValue @ torch.conj(B11MidValue).T).T
#
#
#         T2 = ((A21 * B21+A22 * B22))
#         # print(theta.shape)
#         # print(torch.diag(theta).shape)
#         # print(torch.conj(theta[:,None]).T.shape)
#         R1 = torch.conj(theta[:,None]).T @ T1 @ theta[:,None] - ( 2 * torch.real( torch.conj(theta[:,None]).T @ b1))
#         R2 = torch.conj(theta[:,None]).T @ T2 @ theta[:,None] - ( 2 * torch.real( torch.conj(theta[:,None]).T @ b2))
#         R = torch.max(torch.real(R1),torch.real(R2))
#
#         return  R
#
#     def updateMediumValue(self, phaseRIS, F1, F2, Hr_rx1, Hr_rx2, Hr_tx):
#
#         Theta = torch.diag(phaseRIS)
#         U1 = (Hr_rx1 @ Theta @ Hr_tx) @ F1
#         U2 = (Hr_rx2 @ Theta @ Hr_tx) @ F2
#
#         N = (Hr_rx1).shape[0]
#         I_N = torch.eye(N, N)
#         mu1MidValue = Hr_rx1 @ Theta @ Hr_tx @ F2
#         Mu1 = mu1MidValue @ torch.conj(mu1MidValue).T + self.noise * I_N
#         mu2MidValue = Hr_rx2 @ Theta @ Hr_tx @ F1
#         Mu2 = mu2MidValue @ torch.conj(mu2MidValue).T + self.noise * I_N
#
#         W1 = torch.conj(U1).T @ torch.linalg.inv(( torch.conj(U1@U1).T + Mu1))
#         W2 = torch.conj(U2).T @ torch.linalg.inv(( torch.conj(U2@U2).T + Mu2))
#
#         Sigma1 = I_N - W1 @ U1
#         Sigma2 = I_N - W2 @ U2
#
#         return W1, W2, Sigma1, Sigma2
#
#     def forward(self, theta1, G1, G2, U):
#         M1 = self.N_t
#         N1 = self.N_r
#
#         F1_test_real = theta1[:, 200:200 + M1 * N1]
#         F1_test_imag = theta1[:, 200 + M1 * N1:200 + 2 * M1 * N1]
#
#
#
#         F2_test_real = theta1[:, 200 + 2 * M1 * N1:200 + 3 * M1 * N1]
#         F2_test_imag = theta1[:, 200 + 3 * M1 * N1:]
#
#         F1 = F1_test_real + 1j*F1_test_imag
#         F2 = F2_test_real + 1J*F2_test_imag
#
#         F1 = F1[:,:,None]
#         F2 = F2[:,:,None]
#
#         F1 = F1 * torch.sqrt(2 / (torch.transpose(torch.conj(F1), 1, 2) @ F1))
#         F2 = F2 * torch.sqrt(2 / (torch.transpose(torch.conj(F2), 1, 2) @ F2))
#
#
#
#         theta_real = theta1[:, 0:100]
#         theta_imag = theta1[:, 100:200]
#
#         theta = theta_real + 1j* theta_imag
#
#
#         Nsample = G1.shape[0]
#
#         # R = 0
#
#         for i in range(Nsample):
#             W1, W2, Sigma1, Sigma2 = self.updateMediumValue(theta[i,:],torch.reshape(F1[i,:,:],[self.N_t,self.N_r]),torch.reshape(F2[i,:,:],[self.N_t,self.N_r]),G1[i,:,:],G2[i,:,:],U[i,:,:])
#             R = self.updatePhaseRIS(W1,W2,Sigma1,Sigma2,torch.reshape(F1[i,:,:],[self.N_t,self.N_r]),torch.reshape(F2[i,:,:],[self.N_t,self.N_r]),theta[i,:],G1[i,:,:],G2[i,:,:],U[i,:,:])
#
#
#         return R/Nsample
#


class MyLossFunction(torch.nn.Module):
    def __init__(self, noise, N_t, N_r):
        super(MyLossFunction, self).__init__()
        self.noise = noise
        self.N_t = N_t
        self.N_r = N_r

    def forward(self,theta1, G1, G2, U):

        M1 = self.N_t
        N1 = self.N_r

        F1_test_real = theta1[:, 200:200 + M1 * N1]
        F1_test_imag = theta1[:, 200 + M1 * N1:200 + 2 * M1 * N1]



        F2_test_real = theta1[:, 200 + 2 * M1 * N1:200 + 3 * M1 * N1]
        F2_test_imag = theta1[:, 200 + 3 * M1 * N1:]

        F1 = F1_test_real + 1j*F1_test_imag
        F2 = F2_test_real + 1J*F2_test_imag

        F1 = F1[:,:,None]
        F2 = F2[:,:,None]

        F1 = F1 * torch.sqrt(2 / (torch.transpose(torch.conj(F1), 1, 2) @ F1))
        F2 = F2 * torch.sqrt(2 / (torch.transpose(torch.conj(F2), 1, 2) @ F2))



        theta_real = theta1[:, 0:100]
        theta_imag = theta1[:, 100:200]

        theta = theta_real + 1j* theta_imag


        Nsample = G1.shape[0]

        R = 0

        for i in range(Nsample):

            phi1 = G1[i,:,:] @ torch.diag(theta[i,:]) @ U[i,:,:]
            phi2 = G2[i,:,:] @ torch.diag(theta[i,:]) @ U[i,:,:]

            Mu1 = phi1 @ torch.reshape(F2[i,:,:],[self.N_t,self.N_r]) @ torch.conj(phi1 @ torch.reshape(F2[i,:,:],[self.N_t,self.N_r])).T + self.noise
            Mu2 = phi2 @ torch.reshape(F1[i,:,:],[self.N_t,self.N_r]) @ torch.conj(phi2 @ torch.reshape(F1[i,:,:],[self.N_t,self.N_r])).T + self.noise

            print(Mu1)
            print("theta==>", theta[i,:])

            R1 = torch.real(torch.log(torch.linalg.det(1 + torch.conj(phi1 @ torch.reshape(F1[i,:,:],[self.N_t,self.N_r])).T @ torch.linalg.inv(Mu1) @ phi1 @ torch.reshape(F1[i,:,:],[self.N_t,self.N_r]))))
            R2 = torch.real(torch.log(torch.linalg.det(1 + torch.conj(phi2 @ torch.reshape(F2[i,:,:],[self.N_t,self.N_r])).T @ torch.linalg.inv(Mu2) @ phi2 @ torch.reshape(F2[i,:,:],[self.N_t,self.N_r]))))


            R += max(-R1,-R2)

        return torch.sum(R) / Nsample

# Pt = 2
# F1 = torch.randn([N_sample_train, N_t, N_r]) + 1j*torch.randn([N_sample_train, N_t, N_r])
# F2 = torch.randn([N_sample_train, N_t, N_r]) + 1j*torch.randn([N_sample_train, N_t, N_r])
#
#
# for iii in range(N_sample_train):
#
#     F1[iii,:,:] = F1[iii,:,:] / np.linalg.norm(F1[iii,:,:], 'fro') * np.sqrt((Pt))
#     F2[iii,:,:] = F2[iii,:,:] / np.linalg.norm(F2[iii,:,:], 'fro') * np.sqrt((Pt))

# define the iterator
def data_iter(batch_size, x_train, x_train_original, U, G1, G2, N_sample_train):
    indices = list(range(N_sample_train))
    for i in range(0, N_sample_train, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, N_sample_train)])  # 最后一次可能不足一个batch
        j = j.to(device)
        yield x_train.index_select(0, j), x_train_original.index_select(0,j), U.index_select(0,j), G1.index_select(0,j), G2.index_select(0,j)


tx_antenna_gain = 2
rx_antenna_gain = 2

U = np.zeros([1,N_ris,N_t], dtype=np.complex128)
G1 = np.zeros([1,N_r,N_ris], dtype=np.complex128)
G2 = np.zeros([1,N_r,N_ris], dtype=np.complex128)
# U, G1, G2 =  generalState.generateAllChannel(x_train_original)
myChannel = MyFadingChannel.MyRacianFading(f,tx_antenna_gain,rx_antenna_gain,N_t,N_r,N_ris,K,3,x_train_original[0,0].numpy(),x_train_original[0,1].numpy(),x_train_original[0,2].numpy(),D,dist_ris)
Hdir1, Hdir2, U[0,:,:], G1[0,:,:], G2[0,:,:] =   myChannel.generateAllChannel()
U = torch.from_numpy(U).cfloat()
G1 = torch.from_numpy(G1).cfloat()
G2 = torch.from_numpy(G2).cfloat()

U = U.to(device)
G1 = G1.to(device)
G2 = G2.to(device)

# neural network model
my_model = DNN_Model(N_ris*2, N_r, N_t, D, K, f, dist_ris, threshold)
# my_model.load_state_dict(torch.load("../input/parameters/parameters.pkl"))

my_model = my_model.to(device)
# my_model.load_state_dict(torch.load("parameters.pkl"))

# loss function
my_loss = MyLossFunction(noise, N_t, N_r)
# my_loss = nn.MSELoss()
# my_loss = my_loss.to(device)
#
# loss = nn.L1Loss()
# loss = loss.to(device)

x_train = x_train.to(device)
x_train_original = x_train_original.to(device)


# N_sample_train = torch.tensor(N_sample_train).to(device)

#
# optimizer
optimizer = torch.optim.Adagrad(my_model.parameters(), lr=0.005)
#
# lambda_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200,500,1000,1500], gamma=0.5)
batch_size = 32
# start to train our model

epoch = 50

loss_list = np.zeros([1, int(epoch)])
loss_list_test = np.zeros([1, int(epoch)])

#
for epo in range(epoch):
    # print("epoch=",epo)

    # batch = 0
    if epo%100==0:
        print("training >>>>>>>>>>>>>>", epo)

    for x_train_batch,x_train_original_batch, U_batch, G1_batch, G2_batch in data_iter(batch_size, x_train, x_train_original, U, G1, G2, N_sample_train):
        output = my_model(x_train_batch.float(), x_train_original_batch, U_batch)
        l = my_loss((output).float(),G1_batch,G2_batch,U_batch)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    print("train_loss:==>", l)
    loss_list[0, int(epo)] = l.to(device_cpu).detach().numpy()


torch.save(my_model.state_dict(),"parameters.pkl")

scio.savemat("loss_list.mat",{"loss_list":loss_list})
scio.savemat("loss_list_test.mat",{"loss_list_test":loss_list_test})

my_model.load_state_dict(torch.load("parameters.pkl"))



M1 = N_t
N1 = N_r

y_test_matlab = torch.zeros(10, 10, N_sample_train, dtype=torch.complex64)
F1_test_matlab = torch.zeros(M1, N1, N_sample_train, dtype=torch.complex64)
F2_test_matlab = torch.zeros(M1, N1, N_sample_train, dtype=torch.complex64)



y_test_cc = output.to(device_cpu)

y_test_real = y_test_cc[:,0:100]
y_test_imag = y_test_cc[:,100:200]

y_test = y_test_real+1j*y_test_imag
y_test = torch.reshape(y_test,[N_sample_train,10,10])



F1_test_real = y_test_cc[:,200:200+M1*N1]
F1_test_imag = y_test_cc[:,200+M1*N1:200+2*M1*N1]

F1_test = F1_test_real + 1j*F1_test_imag
F1_test = torch.reshape(F1_test, [N_sample_train,M1,N1])


F2_test_real = y_test_cc[:,200+2*M1*N1:200+3*M1*N1]
F2_test_imag = y_test_cc[:,200+3*M1*N1:]

F2_test = F2_test_real + 1j*F2_test_imag
F2_test = torch.reshape(F2_test, [N_sample_train,M1,N1])

# F1_test = F1
# F2_test = F2

#
for i in range(N_sample_train):
    y_test_matlab[:,:,i] = y_test[i,:,:]
    F1_test_matlab[:,:,i] = F1_test[i,:,:]
    F2_test_matlab[:, :, i] = F2_test[i, :, :]




scio.savemat("y_test_DNN_case1.mat", {"y_test_DNN_case1": y_test_matlab.detach().numpy()})
scio.savemat("F1_test.mat", {"F1_test": F1_test_matlab.detach().numpy()})
scio.savemat("F2_test.mat", {"F2_test": F2_test_matlab.detach().numpy()})
scio.savemat("x_test.mat", {"x_test": x_train_original.detach().numpy()})