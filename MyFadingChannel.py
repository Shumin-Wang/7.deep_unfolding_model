import numpy as np


class RacianFading:
    def __init__(self, frequency, tx_antenna_gain, rx_antenna_gain, Nt, Nr, Nris, K, PLExponent):
        self.frequency = frequency
        self.tx_antenna_gain = tx_antenna_gain
        self.rx_antenna_gain = rx_antenna_gain
        self.Nt = Nt
        self.Nr = Nr
        self.Nris = Nris
        self.K = K
        self.lambdaV = 3e8 / frequency
        self.wn = 2 * np.pi / self.lambdaV
        self.PLExponent = PLExponent

    def generateFSPLDir(self, PLExponent, dist_mid_tx_rx):
        return (4 * np.pi/self.lambdaV)** 2 * (dist_mid_tx_rx**PLExponent)

    def generateHLos(self, D_each_element):
        return np.exp(1j*self.wn*D_each_element)

    def generateModel(self, HLos):
        HNLos = np.sqrt(1 / 2) * (np.random.randn(np.shape(HLos)[0], np.shape(HLos)[1]) + 1j * np.random.randn(np.shape(HLos)[0], np.shape(HLos)[1]))
        H = (1 / np.sqrt(self.K + 1)) * (np.sqrt(self.K) * HLos + HNLos * HLos)
        return H

class typicalCommSym(RacianFading):
    def __init__(self, frequency, tx_antenna_gain, rx_antenna_gain, Nt, Nr, Nris, K, PLExponent):
        RacianFading.__init__(self, frequency, tx_antenna_gain, rx_antenna_gain, Nt, Nr, Nris, K, PLExponent)
        self.s_ris = self.lambdaV /2
        self.dt = self.lambdaV /2
        self.dr = self.lambdaV /2
        self.constant = np.sqrt((self.lambdaV ** 4) / (256 * (np.pi ** 2)))


    def generatePositionTxArr(self,lt):
        # x, y and z axis
        # TX array
        tx_arr = np.zeros([3, self.Nt])
        tx_arr[0,:] = -self.dist
        tx_arr[2,:] = lt + (np.linspace(1,self.Nt,self.Nt) * self.dt - (self.Nt + 1) * self.dt / 2)

        return tx_arr

    def generatePositionRxArr(self,lr):
        rx_arr = np.zeros([3, self.Nr])
        rx_arr[0,:] = self.D - self.dist
        rx_arr[2,:] =  lr + (np.linspace(1,self.Nr,self.Nr) * self.dr - (self.Nr + 1) * self.dr / 2)
        return rx_arr

    def generatePositionRISArr(self):
        ris_arr = np.zeros([3, self.Nris])
        for i in range(int(np.sqrt(self.Nris))):
            for ii in range(int(np.sqrt(self.Nris))):
                ris_arr[0, (i ) * int(np.sqrt(self.Nris)) + ii] = (i + 1 - (np.sqrt(self.Nris) + 1) / 2) * self.s_ris
                ris_arr[1, (i ) * int(np.sqrt(self.Nris)) + ii] = (ii + 1 - (np.sqrt(self.Nris) + 1) / 2) * self.s_ris
        return ris_arr

    def generateDistTXRIS(self,lt):
        tx_arr = self.generatePositionTxArr(lt)
        ris_arr = self.generatePositionRISArr()
        dist_tx_ris = np.zeros([self.Nt, self.Nris])
        for i in range(self.Nt):
            for ii in range(self.Nris):
                dist_tx_ris[i, ii] = np.linalg.norm(tx_arr[:, i] - ris_arr[:, ii])
        dist_tx_ris = dist_tx_ris.T

        return dist_tx_ris

    def generateDistTXRX(self, lt, lr):
        dist_tx_rx = np.zeros([self.Nr, self.Nt])
        tx_arr = self.generatePositionTxArr(lt)
        rx_arr = self.generatePositionRxArr(lr)
        for i1 in range(self.Nr):
            for j1 in range(self.Nt):
                dist_tx_rx[i1, j1] = np.linalg.norm(rx_arr[:, i1]-tx_arr[:, j1])
        return dist_tx_rx

    def generateDistRXRIS(self,lr):
        rx_arr = self.generatePositionRxArr(lr)
        ris_arr = self.generatePositionRISArr()
        dist_rx_ris = np.zeros([self.Nr, self.Nris])
        for i in range(self.Nr):
            for ii in range(self.Nris):
                dist_rx_ris[i, ii] = np.linalg.norm(rx_arr[:, i]-ris_arr[:, ii])

        return dist_rx_ris

    def generateChannelTXRX(self,lt,lr, dist_mid_tx_rx):
        dist_tx_rx = self.generateDistTXRX(lt, lr)
        FSPLDir = self.generateFSPLDir(self.PLExponent, dist_mid_tx_rx)
        Hdir = np.sqrt(1 / FSPLDir) * self.generateModel(dist_tx_rx)

        return Hdir

    def generateChannelTXRIS(self, tx_arr, ris_arr, Nt):
        Hr_tx_los = np.zeros([Nt, self.Nris],dtype=np.complex)
        for i in range(Nt):
            for ii in range(self.Nris):
                distance_nt_nm = np.linalg.norm(tx_arr[:, i] - ris_arr[:, ii] )
                cos_theta_t_nt_nm = tx_arr[2, i] / distance_nt_nm
                Amplitude_nt_nm = self.constant * self.tx_antenna_gain * cos_theta_t_nt_nm / distance_nt_nm / distance_nt_nm
                Phase_term_nr_nm = np.exp(1j * 2 * np.pi * distance_nt_nm / self.lambdaV)
                Hr_tx_los[i, ii] = np.sqrt(Amplitude_nt_nm) * Phase_term_nr_nm
        Hr_tx = self.generateModel(Hr_tx_los)
        Hr_tx = np.conj(Hr_tx).T
        return Hr_tx

    def generateChannelRXRIS(self,rx_arr, ris_arr, Nr):
        Hr_rx_los = np.zeros([Nr, self.Nris], dtype = np.complex)
        for i in range(Nr):
            for ii in range(self.Nris):
                distance_nr_nm = np.linalg.norm(rx_arr[:, i] - ris_arr[:, ii])
                cos_theta_r_nr_nm = rx_arr[2, i] / distance_nr_nm
                Amplitude_nr_nm = self.constant * self.rx_antenna_gain * cos_theta_r_nr_nm / distance_nr_nm / distance_nr_nm
                Phase_term_nr_nm = np.exp(1j * 2 * np.pi * distance_nr_nm / self.lambdaV)
                Hr_rx_los[i, ii] = np.sqrt(Amplitude_nr_nm) * Phase_term_nr_nm
        Hr_rx = self.generateModel(Hr_rx_los)
        return Hr_rx

    def generateChannelRXRISForConstrain(self,rx_arr):
        ris_arr = self.generatePositionRISArr()
        Hr_rx_los = np.zeros([self.Nr, self.Nris])
        for i in range(self.Nr):
            for ii in range(self.Nris):
                distance_nr_nm = np.norm(rx_arr[:, i] - ris_arr[:, ii] )
                cos_theta_r_nr_nm = rx_arr[2, i] / distance_nr_nm
                Amplitude_nr_nm = self.constant * self.rx_antenna_gain * cos_theta_r_nr_nm / distance_nr_nm / distance_nr_nm
                Phase_term_nr_nm = np.exp(1j * 2 * np.pi * distance_nr_nm / self.lambdaV)
                Hr_rx_los[i, ii] = np.sqrt(Amplitude_nr_nm) * Phase_term_nr_nm
        Hr_rx = self.generateModel(Hr_rx_los)
        return Hr_rx


class MyRacianFading(typicalCommSym):
    def __init__(self,frequency, tx_antenna_gain, rx_antenna_gain, Nt, Nr, Nris, K, PLExponent, theta_t, theta_r1, theta_r2, D, dist):
        typicalCommSym.__init__(self,frequency, tx_antenna_gain, rx_antenna_gain, Nt, Nr, Nris, K, PLExponent)
        self.theta_r1 = theta_r1
        self.theta_r2 = theta_r2
        self.theta_t = theta_t
        self.D = D
        self.dist = dist

        self.lt = dist / np.tan(theta_t*np.pi/180)
        self.lr1 = (D - dist) / np.cos(theta_r1/180*np.pi)
        self.lr2 = (D - dist) / np.cos(theta_r2/180*np.pi)
        self.lr = self.lr1 / 2 + self.lr2 / 2

        self.dist_mid_tx_rx1 = np.sqrt((self.dist + (self.D - self.dist) * np.sin(theta_r1/180*np.pi)) ** 2 + (self.lt - self.lr1) ** 2)
        self.dist_mid_tx_rx2 = np.sqrt((self.dist + (self.D - self.dist) * np.sin(theta_r1/180*np.pi)) ** 2 + (self.lt - self.lr2) ** 2)

    def generateAllChannel(self):
        dist_tx_rx1 = self.generateDistTXRX(self.lt, self.theta_r1)
        dist_tx_rx2 = self.generateDistTXRX(self.lt, self.theta_r2)

        FSPLDir1 = self.generateFSPLDir(self.PLExponent, self.dist_mid_tx_rx1)
        FSPLDir2 = self.generateFSPLDir(self.PLExponent, self.dist_mid_tx_rx2)

        Hdir1 = np.sqrt(1 / FSPLDir1) * self.generateModel(dist_tx_rx1)
        Hdir2 = np.sqrt(1 / FSPLDir2) * self.generateModel(dist_tx_rx2)

        # % rx1_arr = obj.generatePositionRxArr(obj.lr1, obj.theta_r1);
        tx_arr = self.generatePositionTxArr(self.lt)
        rx_arr1 = self.generatePositionRxArr(self.theta_r1)
        rx_arr2 = self.generatePositionRxArr(self.theta_r2)
        ris_arr = self.generatePositionRISArr()

        # % rx2_arr = obj.generatePositionRxArr(obj.lr2, obj.theta_r2);
        Hr_tx = self.generateChannelTXRIS(tx_arr, ris_arr, self.Nt)
        Hr_rx1 = self.generateChannelRXRIS(rx_arr1, ris_arr, self.Nr)
        Hr_rx2 = self.generateChannelRXRIS(rx_arr2, ris_arr, self.Nr)

        return Hdir1, Hdir2, Hr_tx, Hr_rx1, Hr_rx2


