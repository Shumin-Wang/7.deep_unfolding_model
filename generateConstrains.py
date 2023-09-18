import numpy as np
import torch


def generate_constrainsChannleAll(dd, D2, Nt, Nr, Nris, lt, D, K, f, dist_ris,ob_down,ob_up):
    [phi01, theta01] = np.meshgrid(0, np.arange(ob_down, ob_up + dd, dd))
    z_matrix1 = D2 * np.cos(theta01 / 180 * np.pi)  # convert to 3D Cartesian coordinate system (position matrix of Rx antenna)
    x_matrix1 = D2 * np.sin(theta01 / 180 * np.pi) * np.cos(phi01 / 180 * np.pi)
    y_matrix1 = D2 * np.sin(theta01 / 180 * np.pi) * np.sin(phi01 / 180 * np.pi)

    rx_arr = np.zeros([3, Nr])
    G_list = np.zeros([len(theta01), Nr, Nris], dtype=np.complex128)
    for iii in range(len(theta01)):
        rx_arr[0, :] = x_matrix1[iii]
        rx_arr[1, :] = y_matrix1[iii]
        rx_arr[2, :] = z_matrix1[iii]
        [Hdir3, U, G3] = chan_mat_RIS_new_model_test(Nt, Nr, Nris, lt, z_matrix1[iii], D, 1, K, f, dist_ris, rx_arr)
        G_list[iii, :, :] = G3

    return G_list

def generate_constrains_mid(F1, F2, dd, D2, Nt, Nr, Nris, lt, D, K, f, dist_ris, ob_down, ob_up,U):

    F1 = F1.detach().numpy()
    F2 = F2.detach().numpy()

    G_list= generate_constrainsChannleAll(dd, D2, Nt, Nr, Nris, lt, D, K, f, dist_ris, ob_down, ob_up)


    # G_list = torch.from_numpy(G_list)


    A_list = np.conj(G_list).transpose(0,2,1) @ G_list

    B_list_mid = U @ F1



    B_list =(B_list_mid @ (np.conj(B_list_mid)).T).T


    C_list_mid = U @ F2
    C_list = (C_list_mid @ (np.conj(C_list_mid)).T).T

    T_list =  (A_list * B_list + A_list * C_list)

    return T_list


def generate_constrains(theta_r1, theta_r2, F1, F2, dd, D2, Nt, Nr, Nris, lt, D, K, f, dist_ris,U):
    U = U.numpy()
    T_list1 = generate_constrains_mid(F1, F2, dd, D2, Nt, Nr, Nris, lt, D, K, f, dist_ris, -89, theta_r1-10,U)
    T_list3 = generate_constrains_mid(F1, F2, dd, D2, Nt, Nr, Nris, lt, D, K, f, dist_ris, theta_r1+10, 89,U)
    if  theta_r2 - theta_r1 >= 20:
            T_list2 = generate_constrains_mid(F1, F2, dd, D2, Nt, Nr, Nris, lt, D, K, f, dist_ris, theta_r1+10, theta_r2-10,U)
            T_list = np.concatenate([T_list1, T_list2, T_list3], 0)
    else:
        T_list = np.concatenate([T_list1, T_list3], 0)

    T_list = torch.from_numpy(T_list).cfloat()

    return T_list


# def generate_constrainsChannleAll(dd, D2, Nt, Nr, Nris, lt, D, K, f, dist_ris):
#
#         [phi01, theta01] = np.meshgrid(0, np.arange(-89, 89 + dd, dd))
#
#
#         z_matrix1 = D2 * np.cos(
#             theta01 / 180 * np.pi)  # convert to 3D Cartesian coordinate system (position matrix of Rx antenna)
#         x_matrix1 = D2 * np.sin(theta01 / 180 * np.pi) * np.cos(phi01 / 180 * np.pi)
#         y_matrix1 = D2 * np.sin(theta01 / 180 * np.pi) * np.sin(phi01 / 180 * np.pi)
#
#         rx_arr = np.zeros([3, Nr])
#         G_list = np.zeros([len(theta01), Nr, Nris], dtype=np.complex128)
#         for iii in range(len(theta01)):
#             rx_arr[0, :] = x_matrix1[iii]
#             rx_arr[1, :] = y_matrix1[iii]
#             rx_arr[2, :] = z_matrix1[iii]
#             [Hdir3, U, G3] = chan_mat_RIS_new_model_test(Nt, Nr, Nris, lt, z_matrix1[iii], D, 1, K, f, dist_ris, rx_arr)
#             G_list[iii, :, :] = G3
#
#         return G_list, np.squeeze(theta01),U
#
#
#
#
# def generate_constrains(theta_r1, theta_r2, F1, F2, U, G_list_All,theta_all,Pt):
#
#     theta_r1 = theta_r1.numpy()
#     theta_r2 = theta_r2.numpy()
#     F1 = F1.detach().numpy()
#     F2 = F2.detach().numpy()
#
#     # F1 = F1 / np.linalg.norm(F1, 'fro') * np.sqrt((Pt))
#     # F2 = F2 / np.linalg.norm(F2, 'fro') * np.sqrt((Pt))
#
#
#     if  theta_r2 - theta_r1 >= 20:
#         index111 = np.where(((theta_all >= -89) & (theta_all <= theta_r1 - 10)) | ((theta_all >= theta_r2 + 10) & (theta_all <= 89)) | ((theta_all >=theta_r1 + 10) & (theta_all <= theta_r2 - 10)))
#     else:
#         index111 = np.where(((theta_all >= -89) & (theta_all <= theta_r1 - 10)) | ((theta_all >= theta_r2 + 10) & (theta_all <= 89)))
#
#
#     G_list = G_list_All[np.squeeze(index111),:,:]
#
#     # G_list = torch.from_numpy(G_list)
#
#
#     A_list = np.conj(G_list).transpose(0,2,1) @ G_list
#
#     B_list_mid = U @ F1
#
#
#
#     B_list =(B_list_mid @ (np.conj(B_list_mid)).T).T
#
#
#     C_list_mid = U @ F2
#     C_list = (C_list_mid @ (np.conj(C_list_mid)).T).T
#
#     T_list =  (A_list * B_list + A_list * C_list)
#     T_list = torch.from_numpy(T_list).cfloat()
#
#     return T_list








def chan_mat_RIS_new_model_test(Nt,Nr,N_ris,lt,lr,D,no_mat,K,f,dist_ris,rx_arr):
    lambdad = 3e8 / f # wavelength
    dt = lambdad / 2 # TX antenna space
    dr = lambdad / 2 # RX antenna space
    dris = lambdad / 2 # RIS element space
    k = 2 * np.pi / lambdad # wavenumber


    Gt=2 # Tx antenna gain, not in dB, 20.4dB in datasheet
    Gr=2 # Rx antenna gain, not in dB, 20.4dB in datasheet
    G = 1

    tx_arr = np.zeros([3, Nt])
    tx_arr[0,:] = -dist_ris
    tx_arr[2,:] = lt + (np.arange(1, Nt+1, 1) * dt - (Nt + 1) * dt / 2)

    RISPosition = np.zeros([3, N_ris])

    for i in range(int(np.sqrt(N_ris))):
        for ii in range(int(np.sqrt(N_ris))):
            RISPosition[0, (i) * int(np.sqrt(N_ris)) + ii] = (i + 1 - (np.sqrt(N_ris) + 1) / 2) * dris
            RISPosition[1, (i) * int(np.sqrt(N_ris)) + ii] = (ii + 1 - (np.sqrt(N_ris) + 1) / 2) * dris

    Constant1 = np.sqrt(lambdad ** 4 / (256 * np.pi ** 2))
    H1_los = np.zeros([N_ris, Nt], dtype = np.complex128)
    for i in range(Nt):
        for ii in range(N_ris):
            distance_nt_nm = np.linalg.norm(tx_arr[:, i] - RISPosition[:, ii])
            cos_theta_t_nt_nm = tx_arr[2, i] / distance_nt_nm
            Amplitude_nt_nm = Constant1 * G * Gt * cos_theta_t_nt_nm / distance_nt_nm / distance_nt_nm
            Phase_term_nt_nm = np.exp(1j * 2 * np.pi * distance_nt_nm /lambdad)
            H1_los[ii, i] = np.sqrt(Amplitude_nt_nm) * Phase_term_nt_nm
    H2_los = np.zeros([Nr, N_ris], dtype = np.complex128)
    for i in range(Nr):
        for ii in range(N_ris):
            distance_nr_nm =  np.linalg.norm(rx_arr[:, i] - RISPosition[:, ii] )
            cos_theta_r_nr_nm = rx_arr[2, i] / distance_nr_nm
            Amplitude_nr_nm = Constant1 * G * Gr * cos_theta_r_nr_nm / distance_nr_nm / distance_nr_nm
            Phase_term_nr_nm = np.exp(1j * 2 * np.pi * distance_nr_nm /lambdad )
            H2_los[i, ii] = np.sqrt(Amplitude_nr_nm) * Phase_term_nr_nm

    d = np.zeros([Nr, Nt])
    for i1 in range(Nr):
        for j1 in range(Nt):
            d[i1, j1] = np.linalg.norm(rx_arr[:, i1]-tx_arr[:, j1])

    Hdir_los = np.exp(1j * k * d)*np.sqrt((lambdad /(4 * np.pi)) ** 2 / (d** (3)))
    tx_rx_dist = np.sqrt(D ** 2 + (lt - lr) ** 2)
    # FSPL_dir = (lambdad /(4 * np.pi)) ** 2 / tx_rx_dist ** 3
    # Hdir_los = np.exp(1j * k * d)*np.sqrt((lambdad /(4 * np.pi)) ** 2 / (d**(3)))
    # tx_rx_dist = np.sqrt(D ** 2 + (lt - lr) ** 2)
    # FSPL_dir = (lambdad /(4 * np.pi)) ** 2 / tx_rx_dist ** 3

    temp = 1
    Hdir = Rician_ewise(Hdir_los, no_mat, K)
    H1 = Rician_ewise(H1_los * temp, no_mat, K)
    H2 = Rician_ewise(H2_los / temp, no_mat, K)

    return Hdir, H1, H2


def Rician_ewise(Hlos, no_mat, K):

    [M,N] = (np.shape(Hlos))
    Hout = np.zeros([M, N], dtype = np.complex128)

    Hnlos = np.sqrt(1 / 2) * (np.random.randn(M,N) + 1j * np.random.randn(M,N))
    Htot = (Hlos * np.sqrt(K) + Hnlos * Hlos) / np.sqrt(K + 1)
    Hout = Htot
    return Hout


def validate(theta, T_list, threshold):
    value1 = (10 ** ((threshold) / 10)) / 1000
    g_len = len(T_list)

    a_value = np.conj(theta).T @ T_list @ theta
    thd = value1
    c = (a_value <= thd)
    # print(np.sum(c))
    if np.sum(c) == g_len:
        final_c = True
    else:
        final_c = False

    return final_c


def chan_mat_RIS_new_model_ob(Nt,Nr,N_ris,lt,lr,D,no_mat,K,f,dist_ris):
    lambdad = 3e8 / f # wavelength
    dt = lambdad / 2 # TX antenna space
    dr = lambdad / 2 # RX antenna space
    dris = lambdad / 2 # RIS element space
    k = 2 * np.pi / lambdad # wavenumber


    Gt=2 # Tx antenna gain, not in dB, 20.4dB in datasheet
    Gr=2 # Rx antenna gain, not in dB, 20.4dB in datasheet
    G = 1

    tx_arr = np.zeros([3, Nt])
    tx_arr[0,:] = -dist_ris
    tx_arr[2,:] = lt + (np.arange(1, Nt+1, 1) * dt - (Nt + 1) * dt / 2)

    rx_arr = np.zeros([3, Nr])
    rx_arr[0,:] = D - dist_ris
    rx_arr[2,:] =  lr + (np.arange(1,Nr+1, 1) * dr - (Nr + 1) * dr / 2)

    RISPosition = np.zeros([3, N_ris])

    for i in range(int(np.sqrt(N_ris))):
        for ii in range(int(np.sqrt(N_ris))):
            RISPosition[0, i * int(np.sqrt(N_ris)) + ii] = (i + 1 - (np.sqrt(N_ris) + 1) / 2) * dris
            RISPosition[1, i * int(np.sqrt(N_ris)) + ii] = (ii + 1 - (np.sqrt(N_ris) + 1) / 2) * dris

    Constant1 = np.sqrt((lambdad ** 4 / (256 * np.pi ** 2)))
    H1_los = np.zeros([N_ris, Nt], dtype = np.complex128)
    for i in range(Nt):
        for ii in range(N_ris):
            distance_nt_nm = np.linalg.norm(tx_arr[:, i] - RISPosition[:, ii])
            cos_theta_t_nt_nm = tx_arr[2, i] / distance_nt_nm
            Amplitude_nt_nm = Constant1 * G * Gt * cos_theta_t_nt_nm / distance_nt_nm / distance_nt_nm
            Phase_term_nt_nm = np.exp(1j * 2 * np.pi * distance_nt_nm /lambdad)
            H1_los[ii, i] = np.sqrt(Amplitude_nt_nm) * Phase_term_nt_nm


    H2_los = np.zeros([Nr, N_ris], dtype = np.complex128)
    for i in range(Nr):
        for ii in range(N_ris):
            distance_nr_nm =  np.linalg.norm(rx_arr[:, i] - RISPosition[:, ii] )
            cos_theta_r_nr_nm = rx_arr[2, i] / distance_nr_nm
            Amplitude_nr_nm = Constant1 * G * Gr * cos_theta_r_nr_nm / distance_nr_nm / distance_nr_nm
            Phase_term_nr_nm = np.exp(1j * 2 * np.pi * distance_nr_nm /lambdad )
            H2_los[i, ii] = np.sqrt(Amplitude_nr_nm) * Phase_term_nr_nm

    d = np.zeros([Nr, Nt])
    for i1 in range(Nr):
        for j1 in range(Nt):
            d[i1, j1] = np.linalg.norm(rx_arr[:, i1]-tx_arr[:, j1])

    Hdir_los = np.exp(1j * k * d)*np.sqrt((lambdad /(4 * np.pi)) ** 2 / (d** (3)))
    # tx_rx_dist = np.sqrt(D ** 2 + (lt - lr) ** 2)
    # FSPL_dir = (lambdad /(4 * np.pi)) ** 2 / tx_rx_dist ** 3
    # Hdir_los = np.exp(1j * k * d)*np.sqrt((lambdad /(4 * np.pi)) ** 2 / (d**(3)))
    # tx_rx_dist = np.sqrt(D ** 2 + (lt - lr) ** 2)
    # FSPL_dir = (lambdad /(4 * np.pi)) ** 2 / tx_rx_dist ** 3

    temp = 1
    Hdir = Rician_ewise(Hdir_los, no_mat, K)
    H1 = Rician_ewise(H1_los * temp, no_mat, K)
    H2 = Rician_ewise(H2_los / temp, no_mat, K)

    return Hdir, H1, H2



#
# if __name__ == '__main__':
#
#
#
#     theta_t = 30
#     theta_r1 = 30
#     theta_r2 = 60
#
#     dd = 0.5
#     Nt = 8
#     Nr = 2
#     Nris = 10*10
#     lt = 0000
#     D = 50
#     K = 100000
#     f = 27e9
#     dist_ris = 20
#
#     D2 = D - dist_ris
#     lt = dist_ris / np.sin(30*np.pi/theta_t)
#
#     F1 = np.random.randn(10,Nt, Nr) + 1j*np.random.randn(10, Nt, Nr)
#     F2 = np.random.randn(10,Nt, Nr) +1j*np.random.randn(10,Nt, Nr)
#
#     Pt = 2
#     # F1 = F1 / np.linalg.norm(F1, 'fro') * np.sqrt((Pt))
#     # F2 = F2 / np.linalg.norm(F2, 'fro') * np.sqrt((Pt))
#
#     G_list_All, theta_all, U  = generate_constrainsChannleAll(dd, D2, Nt, Nr, Nris, lt, D, K, f, dist_ris)
#
#
#     for i in range(10):
#         F1_bitch = F1[i,:,:] / np.linalg.norm(F1[i,:,:], 'fro') * np.sqrt((Pt))
#         F2_bitch = F2[i,:,:] / np.linalg.norm(F2[i,:,:], 'fro') * np.sqrt((Pt))
#
#         T_list = generate_constrains(theta_r1, theta_r2, F1_bitch, F2_bitch, U, G_list_All, theta_all)
#
#         theta = np.random.randn(10,10) + 1j*np.random.randn(10,10)
#
#         theta = np.reshape(theta,[100,1])
#
#         theta = theta/abs(theta)
#
#         threshold = -120
#
#         threshold_w = (10 ** ((threshold) / 10)) / 1000
#
#         CCC = np.conj(theta).T @ (T_list/threshold_w) @ theta
#
#
#
#         CCC = np.real(np.squeeze(CCC))
#
#         print(np.max(CCC))

# theta = torch.randn([10,100,1])+1j*torch.randn([10,100,1])
#
# theta = theta/abs(theta)
#
# print(theta)
