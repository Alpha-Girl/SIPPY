# -*- coding: utf-8 -*-
"""
Created
@author: Giuseppe Armenise
example armax mimo
case 3 outputs x 4 inputs
"""
from __future__ import division

from past.utils import old_div

# Checking path to access other files
try:
    from sippy import *
except ImportError:
    import sys, os

    sys.path.append(os.pardir)
    from sippy import *

import numpy as np
import control.matlab as cnt
from sippy import functionset as fset
import pandas as pd
for train in range(1,4):
    for val in range(1,4):
        if train!=val:
# train=3
# val=2
            data_train=pd.read_csv('D:\\Github\\TCS-H2\\data'+str(train)+'.csv',header=None)
            # print(data_train.shape)
            # 4*2 MIMO system
            # generating transfer functions in z-operator.
            ts = 1.

            NUM11 = [4, 3.3, 0., 0.]
            NUM12 = [10, 0., 0.]
            NUM13 = [7.0, 5.5, 2.2]
            NUM14 = [-0.9, -0.11, 0., 0.]
            DEN1 = [1., -0.3, -0.25, -0.021, 0., 0.]  #
            H1 = [1., 0.85, 0.32, 0., 0., 0.]
            na1 = 3
            nb11 = 2
            nb12 = 1
            nb13 = 3
            nb14 = 2
            th11 = 1
            th12 = 2
            th13 = 2
            th14 = 1
            nc1 = 2
            #
            DEN2 = [1., -0.4, 0., 0., 0.]
            NUM21 = [-85, -57.5, -27.7]
            NUM22 = [71, 12.3]
            NUM23 = [-0.1, 0., 0., 0.]
            NUM24 = [0.994, 0., 0., 0.]
            H2 = [1., 0.4, 0.05, 0., 0.]
            na2 = 1
            nb21 = 3
            nb22 = 2
            nb23 = 1
            nb24 = 1
            th21 = 1
            th22 = 2
            th23 = 0
            th24 = 0
            nc2 = 2
            #
            # time
            tfin = data_train.shape[0]-1
            npts = int(old_div(tfin, ts)) + 1
            Time = np.linspace(0, tfin, npts)

            #INPUT#
            Usim = np.zeros((4, npts))
            Usim_noise = np.zeros((4, npts))
            Usim[0, :] = data_train.iloc[:,1].values
            Usim[1, :] = data_train.iloc[:,2].values
            Usim[2, :] = data_train.iloc[:,3].values
            Usim[3, :] = data_train.iloc[:,4].values

            Ytot = np.zeros((2, npts))

            Ytot[0, :] = data_train.iloc[:,5].values
            Ytot[1, :] = data_train.iloc[:,6].values
            # Ytot[2, :] = (Ytot3 + err_outputH3).squeeze()

            ##identification parameters
            # ordersna = [na1, na2]
            # ordersnb = [[nb11, nb12, nb13, nb14], [nb21, nb22, nb23, nb24]]
            # ordersnc = [nc1, nc2]
            # theta_list = [[th11, th12, th13, th14], [th21, th22, th23, th24]]
            ordersna = [na1, na1]
            ordersnb = [[nb11, nb12, nb13, nb14], [nb11, nb12, nb13, nb14]]
            ordersnc = [nc1, nc1]
            theta_list = [[th11, th12, th13, th14], [th11, th12, th13, th14]]

            # IDENTIFICATION STAGE
            # TESTING ARMAX models
            # iterative LLS
            Id_ARMAXi = system_identification(Ytot, Usim, 'ARMAX', ARMAX_orders=[ordersna, ordersnb, ordersnc, theta_list], 
                                            max_iterations=20, centering = 'MeanVal')  #
            # optimization-based
            Id_ARMAXo = system_identification(Ytot, Usim, 'ARMAX', ARMAX_orders=[ordersna, ordersnb, ordersnc, theta_list], 
                                            ARMAX_mod = 'OPT', max_iterations=20, centering = 'None')  #
            # recursive LLS
            Id_ARMAXr = system_identification(Ytot, Usim, 'ARMAX', ARMAX_orders=[ordersna, ordersnb, ordersnc, theta_list], 
                                            ARMAX_mod = 'RLLS', max_iterations=20, centering = 'InitVal')  #

            # output of the identified model
            Yout_ARMAXi = Id_ARMAXi.Yid
            Yout_ARMAXo = Id_ARMAXo.Yid
            Yout_ARMAXr = Id_ARMAXr.Yid

            ######plots
            #   
            import matplotlib.pyplot as plt

            # U
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['figure.figsize']=(12.8, 7.2)
            plt.close('all')
            plt.figure(0)
            plt.subplot(4, 1, 1)
            plt.plot(Time, Usim[0, :])
            plt.grid()
            plt.ylabel("Input 1 - TCS流量(KG/h)")
            plt.xlabel("Time")
            plt.title("训练数据炉次"+str(train))

            plt.subplot(4, 1, 2)
            plt.plot(Time, Usim[1, :])
            plt.grid()
            plt.ylabel("Input 2 - H2流量")
            plt.xlabel("Time")

            plt.subplot(4, 1, 3)
            plt.plot(Time, Usim[2, :])
            plt.ylabel("Input 3 - 外圈电流")
            plt.xlabel("Time")
            plt.grid()

            plt.subplot(4, 1, 4)
            plt.plot(Time, Usim[3, :])
            plt.ylabel("Input 4 - 内圈电流")
            plt.xlabel("Time")
            plt.grid()
            plt.savefig(str(train)+"TO"+str(val)+"INPUT_TRAIN.png")
            # Y
            plt.figure(1)
            plt.subplot(2, 1, 1)
            plt.plot(Time, Ytot[0, :])
            plt.plot(Time, Yout_ARMAXi[0,:])
            plt.plot(Time, Yout_ARMAXo[0,:])
            plt.plot(Time, Yout_ARMAXr[0,:])
            plt.ylabel("外圈电压")
            plt.grid()
            plt.xlabel("Time")
            plt.title("训练数据炉次"+str(train))
            plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])

            plt.subplot(2, 1, 2)
            plt.plot(Time, Ytot[1, :])
            plt.plot(Time, Yout_ARMAXi[1,:])
            plt.plot(Time, Yout_ARMAXo[1,:])
            plt.plot(Time, Yout_ARMAXr[1,:])
            plt.ylabel("内圈电压")
            plt.grid()
            plt.xlabel("Time")
            plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])

            plt.savefig(str(train)+"TO"+str(val)+"OUTPUT_TRAIN.png")


            ### VALIDATION STAGE 

            # time
            data_val=pd.read_csv('D:\\Github\\TCS-H2\\data'+str(val)+'.csv',header=None)
            tfin = data_val.shape[0]-1
            npts = int(old_div(tfin, ts)) + 1
            Time = np.linspace(0, tfin, npts)

            # (NEW) INPUTS
            U_valid = np.zeros((4, npts))
            Usim_noise = np.zeros((4, npts))
            U_valid[0, :] = data_val.iloc[:,1].values
            U_valid[1, :] = data_val.iloc[:,2].values
            U_valid[2, :] = data_val.iloc[:,3].values
            U_valid[3, :] = data_val.iloc[:,4].values

            # Total Output

            #
            Ytot_v = np.zeros((2, npts))
            #
            Ytot_v[0, :] = data_val.iloc[:,5].values
            Ytot_v[1, :] = data_val.iloc[:,6].values
            # Ytot_v[2, :] = (Ytot3 + err_outputH3).squeeze()

            # ## Compute time responses for identified systems with new inputs

            # ARMAX - ILLS
            Yv_armaxi = fset.validation(Id_ARMAXi,U_valid,Ytot_v,Time, centering = 'MeanVal')

            # ARMAX - OPT
            Yv_armaxo = fset.validation(Id_ARMAXo,U_valid,Ytot_v,Time)

            # ARMAX - RLLS 
            Yv_armaxr = fset.validation(Id_ARMAXr,U_valid,Ytot_v,Time, centering = 'InitVal')

            # U
            plt.figure(3)
            plt.subplot(4, 1, 1)
            plt.plot(Time, U_valid[0, :])
            plt.grid()
            plt.ylabel("Input 1 - TCS流量(KG/h)")
            plt.xlabel("Time")
            plt.title("验证数据炉次"+str(val))

            plt.subplot(4, 1, 2)
            plt.plot(Time, U_valid[1, :])
            plt.ylabel("Input 2 - H2流量")
            plt.xlabel("Time")
            plt.grid()

            plt.subplot(4, 1, 3)
            plt.plot(Time, U_valid[2, :])
            plt.ylabel("Input 3 - 外圈电流")
            plt.xlabel("Time")
            plt.grid()

            plt.subplot(4, 1, 4)
            plt.plot(Time, U_valid[3, :])
            plt.ylabel("Input 4 - 内圈电流")
            plt.xlabel("Time")
            plt.grid()
            plt.savefig(str(train)+"TO"+str(val)+"INPUT_VAL.png")

            # Y
            plt.figure(4)
            #mean square error
            mse_armaxi = sum((Ytot_v[0, :] - Yv_armaxi[0,:])**2)/len(Ytot_v[0, :])
            mse_armaxo = sum((Ytot_v[0, :] - Yv_armaxo[0,:])**2)/len(Ytot_v[0, :])
            mse_armaxr = sum((Ytot_v[0, :] - Yv_armaxr[0,:])**2)/len(Ytot_v[0, :])
            # min square error
            if mse_armaxi < mse_armaxo:
                if mse_armaxi < mse_armaxr:
                    y=Yv_armaxi[0,:]
                    y_label='ARMAX-I'
                else:
                    if mse_armaxo < mse_armaxr:
                        y=Yv_armaxo[0,:]
                        y_label='ARMAX-O'
                    else:
                        y=Yv_armaxr[0,:]
                        y_label='ARMAX-R'
            plt.subplot(2, 1, 1)
            plt.plot(Time, Ytot_v[0, :])
            # plt.plot(Time, Yv_armaxi[0,:])
            # plt.plot(Time, Yv_armaxo[0,:])
            plt.plot(Time, y)
            plt.ylabel("外圈电压")
            plt.grid()
            plt.xlabel("Time")
            plt.title("验证数据炉次"+str(val))
            # plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])
            plt.legend(['System', y_label])
            mse_armaxi = sum((Ytot_v[1, :] - Yv_armaxi[1,:])**2)/len(Ytot_v[1, :])
            mse_armaxo = sum((Ytot_v[1, :] - Yv_armaxo[1,:])**2)/len(Ytot_v[1, :])
            mse_armaxr = sum((Ytot_v[1, :] - Yv_armaxr[1,:])**2)/len(Ytot_v[1, :])
            # min square error
            if mse_armaxi < mse_armaxo:
                if mse_armaxi < mse_armaxr:
                    y=Yv_armaxi[1,:]
                    y_label='ARMAX-I'
                else:
                    if mse_armaxo < mse_armaxr:
                        y=Yv_armaxo[1,:]
                        y_label='ARMAX-O'
                    else:
                        y=Yv_armaxr[1,:]
                        y_label='ARMAX-R'
            plt.subplot(2, 1, 2)
            plt.plot(Time, Ytot_v[1, :])
            # plt.plot(Time, Yv_armaxi[1,:])
            # plt.plot(Time, Yv_armaxo[1,:])
            plt.plot(Time, y)
            plt.ylabel("内圈电压")
            plt.grid()
            plt.xlabel("Time")
            # plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])
            plt.legend(['System',  y_label])

            plt.savefig(str(train)+"TO"+str(val)+"OUTPUT_VAL.png")

            # plt.show()
