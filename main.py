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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')
print(torch.version.cuda)
print(torch.cuda.is_available())



dd = 0.5
N_t = 8
N_r = 2
N_ris = 10*10
lt = 0000
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
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, N_ris+4*N_r*N_t),
        )




    def forward(self, sample1, sample2):
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

        # F1 = F1 * torch.sqrt(2 / (torch.transpose(torch.conj(F1), 1, 2) @ F1))
        # F2 = F2 * torch.sqrt(2 / (torch.transpose(torch.conj(F2), 1, 2) @ F2))



        theta1[:, 200:200 + M1 * N1] = torch.real(F1.squeeze())
        theta1[:, 200 + M1 * N1:200 + 2 * M1 * N1] = torch.imag(F1.squeeze())

        theta1[:, 200 + 2 * M1 * N1:200 + 3 * M1 * N1] = torch.real(F2.squeeze())
        theta1[:, 200 + 3 * M1 * N1:] = torch.imag(F2.squeeze())

        theta_real = theta1[:, 0:100]
        theta_imag = theta1[:, 100:200]

        theta = theta_real + 1j* theta_imag

        theta = theta/abs(theta)

        theta = self.ProjectRIS(F1,F2,theta,sample2)

        theta1[:, 0:100] = torch.real(theta)
        theta1[:, 100:200] = torch.imag(theta)

        return theta1

    def ProjectRIS(self, F1, F2, theta, sample2):
        theta_hat = torch.zeros([theta.shape[0], theta.shape[1]],dtype=torch.complex128)

        for i in range(theta.shape[0]):

            threshold_w = (10 ** ((self.threshold) / 10)) / 1000

            lt = self.dist_ris / torch.sin(torch.pi*sample2[i,0]/180)
            D2 = self.D - self.dist_ris
            G_list_All, theta_all, U = generateConstrains.generate_constrainsChannleAll(0.5, D2, self.N_t, self.N_r, int(self.N_ris/2), lt, self.D, self.K, self.f, self.dist_ris)
            T_list = generateConstrains.generate_constrains(sample2[i,1], sample2[i,2], torch.reshape(F1[i,:,:],[self.N_t,self.N_r]), torch.reshape(F2[i,:,:],[self.N_t,self.N_r]), U, G_list_All, theta_all,2)
            CCC = torch.conj(theta[i,:,None]).T @ (T_list / threshold_w) @ theta[i,:,None]
            CCC = torch.real(np.squeeze(CCC))
            theta_hat[i,:] = theta[i,:]*torch.sqrt(1/torch.max(CCC))

        return theta_hat



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




            R1 = torch.real(torch.log(torch.linalg.det(1 + torch.conj(phi1 @ torch.reshape(F1[i,:,:],[self.N_t,self.N_r])).T @ torch.linalg.inv(Mu1) @ phi1 @ torch.reshape(F1[i,:,:],[self.N_t,self.N_r]))))
            R2 = torch.real(torch.log(torch.linalg.det(1 + torch.conj(phi2 @ torch.reshape(F2[i,:,:],[self.N_t,self.N_r])).T @ torch.linalg.inv(Mu2) @ phi2 @ torch.reshape(F2[i,:,:],[self.N_t,self.N_r]))))


            R += min(R1,R2)

        return torch.sum(-R) / Nsample


# define the iterator
def data_iter(batch_size, x_train, x_train_original, U, G1, G2, N_sample_train):
    indices = list(range(N_sample_train))
    for i in range(0, N_sample_train, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, N_sample_train)])  # 最后一次可能不足一个batch
        j = j.to(device)
        yield x_train.index_select(0, j), x_train_original.index_select(0,j), U.index_select(0,j), G1.index_select(0,j), G2.index_select(0,j)

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
optimizer = torch.optim.Adagrad(my_model.parameters(), lr=0.01)
#
lambda_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200,500,1000,1500], gamma=0.5)
batch_size = 64
# start to train our model

epoch = 20

loss_list = np.zeros([1, int(epoch)])
loss_list_test = np.zeros([1, int(epoch)])


U, G1, G2 =  generalState.generateAllChannel(x_train_original)

U = torch.from_numpy(U).cfloat()
G1 = torch.from_numpy(G1).cfloat()
G2 = torch.from_numpy(G2).cfloat()

U = U.to(device)
G1 = G1.to(device)
G2 = G2.to(device)
#
for epo in range(epoch):
    # print("epoch=",epo)

    # batch = 0
    if epo%100==0:
        print("training >>>>>>>>>>>>>>", epo)

    for x_train_batch,x_train_original_batch, U_batch, G1_batch, G2_batch in data_iter(batch_size, x_train, x_train_original, U, G1, G2, N_sample_train):
        # batch += 1
        # print("batch_size_times==>",batch)
        output = my_model(x_train_batch.float(), x_train_original_batch)
        l = my_loss((output).float(), G1_batch, G2_batch, U_batch)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()


    y_train_out = my_model(x_train.float(), x_train_original_batch)
    l_train = my_loss((y_train_out).float(), G1_batch, G2_batch, U_batch)
    print("train_loss:==>", l_train)
    loss_list[0, int(epo)] = l_train.to(device_cpu).detach().numpy()

    # y_test_out = my_model(x_test.float())
    # l_test = loss((y_test_out).float(), y_test.float())
    # print("test_loss:==>", l_test)
    # loss_list_test[0, int(epo)] = l_test.to(device_cpu).detach().numpy()



torch.save(my_model.state_dict(),"parameters.pkl")

scio.savemat("loss_list.mat",{"loss_list":loss_list})
scio.savemat("loss_list_test.mat",{"loss_list_test":loss_list_test})

starttime = time.time()
my_model.load_state_dict(torch.load("parameters.pkl"))


# x_test_cc = torch.zeros(6, 3)
#
# x_test_cc[0, :] = torch.tensor([32, 10, 40])
# x_test_cc[1, :] = torch.tensor([31, 15, 45])
# x_test_cc[2, :] = torch.tensor([33, 12, 50])
# x_test_cc[3, :] = torch.tensor([35, 18, 45.6])
# x_test_cc[4, :] = torch.tensor([31.27, 10.5, 45.5])
# x_test_cc[5, :] = torch.tensor([32.5, 12.5, 45.5])

# for i in range(30):
#     x_test_cc[i, :] = torch.tensor([30, 10+i/2, 40])



# x_test_cc[6, :] = torch.tensor([45, 10.7, 30.45])
# x_test_cc[7, :] = torch.tensor([35.3, 36, 33.7])
# x_test_cc[8, :] = torch.tensor([34, 34.3, 54.32])
# x_test_cc[9, :] = torch.tensor([30.5, 10.3, 36.35])
# x_test_cc[10, :] = torch.tensor([35.54, 30.23, 43.5])
# x_test_cc[11, :] = torch.tensor([30.53, 20.54, 46.35])
M1 = N_t
N1 = N_r
#
# x_no_test = x_test_cc
y_test_matlab = torch.zeros(10, 10, N_sample_train, dtype=torch.complex64)
F1_test_matlab = torch.zeros(M1, N1, N_sample_train, dtype=torch.complex64)
F2_test_matlab = torch.zeros(M1, N1, N_sample_train, dtype=torch.complex64)
# # # #
#
#
# x_test_int, x_test_frac = separate_frac_int(x_test_cc)
#
# x_test_int_encoding = enc.transform(x_test_int)
# #
# x_test_cc = processing_train(x_test_int_encoding, x_test_frac)
# x_test_cc = torch.from_numpy(x_test_cc.A)
#
# x_test_cc = x_test_cc.to(device)



# my_model.load_state_dict(torch.load("parameters.pkl"))
# # #
# my_model.eval()
# with torch.no_grad():
#     y_test_cc = my_model(x_test_cc.float())
#
#
#



y_test_cc = output.to(device_cpu)
#
# # y_test_cc = 2*y_test_cc-1
#
# #
y_test_real = y_test_cc[:,0:100]
y_test_imag = y_test_cc[:,100:200]

y_test = y_test_real+1j*y_test_imag
y_test = torch.reshape(y_test,[N_sample_train,10,10])



F1_test_real = y_test_cc[:,200:200+M1*N1]
F1_test_imag = y_test_cc[:,200+M1*N1:200+2*M1*N1]

F1_test = F1_test_real + 1j*F1_test_imag
F1_test = torch.reshape(F1_test, [6,M1,N1])


F2_test_real = y_test_cc[:,200+2*M1*N1:200+3*M1*N1]
F2_test_imag = y_test_cc[:,200+3*M1*N1:]

F2_test = F2_test_real + 1j*F2_test_imag
F2_test = torch.reshape(F2_test, [6,M1,N1])



#
for i in range(6):
    y_test_matlab[:,:,i] = y_test[i,:,:]
    F1_test_matlab[:,:,i] = F1_test[i,:,:]
    F2_test_matlab[:, :, i] = F2_test[i, :, :]

endtime = time.time()
print((endtime - starttime))

#
#
# y1_test = y_test_cc[:, 0:2304]
# y1_test_real = y1_test[:, :24 * 48]
# y1_test_imag = y1_test[:, 24 * 48:]
#
# print(y1_test_real.shape)
#
# y1_test_c = y1_test_real + 1j * y1_test_imag
# y1_test_c = torch.reshape(y1_test_c, [12, 24, 48])
#
# y2_test = y_test_cc[:, 2304:]
# y2_test_real = y2_test[:, :24 * 48]
# y2_test_imag = y2_test[:, 24 * 48:]
#
# y2_test_c = y2_test_real + 1j * y2_test_imag
# y2_test_c = torch.reshape(y2_test_c, [12, 24, 48])
#
# y_test_cc = torch.cat([y1_test_c, y2_test_c], dim=1)

# y_test = y_test_real+1j*y_test_imag
# y_test = torch.reshape(y_test,[12,48,48])

#
# for i in range(12):
#     y_test_matlab[:, :, i] = y_test_cc[i, :, :]
#
# torch.save(my_model.state_dict(),"parameters.pkl")


scio.savemat("y_test_DNN_case1.mat", {"y_test_DNN_case1": y_test_matlab.detach().numpy()})
scio.savemat("F1_test.mat", {"F1_test": F1_test_matlab.detach().numpy()})
scio.savemat("F2_test.mat", {"F2_test": F2_test_matlab.detach().numpy()})
scio.savemat("x_test.mat", {"x_test": x_train_original.detach().numpy()})