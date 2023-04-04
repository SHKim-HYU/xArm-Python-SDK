#!/usr/bin/env python3
from math import sin, cos, tan, pi
from trajectory_generate import Trajectory

import time
import os
import sys
import can
import numpy as np
import matplotlib.pyplot as plt
import modern_robotics as mr

from multiprocessing import Process, Manager


sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI


#######################################################
"""
Just for test example
"""
if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
    except:
        ip = input('Please input the xArm ip address:')
        if not ip:
            print('input error, exit')
            sys.exit(1)
########################################################


#############################################################################
################## Manager for global data multiprocessing ##################
#############################################################################

manager = Manager()
_q = manager.dict()
_q_d = manager.dict()
_F = manager.dict()

_q['q'] = [0.0]*9; _q['q_dot'] = [0.0]*9; _q['trq_g'] = [0.0]*9; _q['mpc_q']=[0.0]*9; _q['mpc_q_dot']=[0.0]*9;
_q['trq_ext'] = [0.0]*6;
_q_d['qd'] = [0.0]*9; _q_d['qd_dot'] = [0.0]*9; _q_d['qd_ddot'] = [0.0]*9;
_F['force'] = [0.0]*3; _F['torque'] = [0.0]*3;

000

def ft_sensor_run():
    init_time=time.time()
    DF=50; DT=2000
    RFT_frq=1000
    channel = 0
    
    bus = can.interface.Bus(bustype = 'kvaser', channel = 0, bitrate = 1000000)

    # Set filter
    tx_message = can.Message(arbitration_id = 0x64, is_extended_id = False, data = [0x08, 0x01, 0x09, 0x01, 0x01, 0x01, 0x01, 0x01])
    bus.send(tx_message, timeout = 0.5)
    bus.recv()

    tx_message = can.Message(arbitration_id = 0x64, is_extended_id = False, data = [0x11, 0x01, 0x09, 0x01, 0x01, 0x01, 0x01, 0x01])
    bus.send(tx_message, timeout = 0.5)
    bus.recv()
    
    while True:
        g_time = time.time()
        # read once
        tx_message = can.Message(arbitration_id = 0x64, is_extended_id = False, data = [0x0A, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01])
        bus.send(tx_message, timeout = 0.5)

        rx_message_1 = bus.recv()
        rx_message_2 = bus.recv()

        fx = ((rx_message_1.data[1]<<8) + rx_message_1.data[2])
        fy = ((rx_message_1.data[3]<<8) + rx_message_1.data[4])
        fz = ((rx_message_1.data[5]<<8) + rx_message_1.data[6])
        signed_fx = (-(fx & 0x8000) | (fx&0x7fff))/DF
        signed_fy = (-(fy & 0x8000) | (fy&0x7fff))/DF
        signed_fz = (-(fz & 0x8000) | (fz&0x7fff))/DF
        
        tx = ((rx_message_1.data[7]<<8) + rx_message_2.data[0])
        ty = ((rx_message_2.data[1]<<8) + rx_message_2.data[2])
        tz = ((rx_message_2.data[3]<<8) + rx_message_2.data[4])
        signed_tx = (-(tx & 0x8000) | (tx&0x7fff))/DT
        signed_ty = (-(ty & 0x8000) | (ty&0x7fff))/DT
        signed_tz = (-(tz & 0x8000) | (tz&0x7fff))/DT
        
        _F['force'] = [signed_fx, signed_fy, signed_fz]
        _F['torque'] = [signed_tx, signed_ty, signed_tz]
        
        while time.time()-g_time<(1/RFT_frq):
            time.sleep(1/1000000)
            glob_time_buf=time.time()
            init_time_buf=init_time
        #print(time.time()-g_time)

def xarm6_cmd_run():
    deg2rad = 3.141592/180
    rad2deg = 180/3.141592
    xarm6_frq = 250
    
    xarm6_mode = 4 #2 #       2: teaching mode, 4: joint velocity control

    init_time=time.time()
    
    qd=[0]*6
    qd_dot=[0]*6
    qd_ddot=[0]*6
    cmd_vel=[0]*6
    
    Js_dot = np.zeros((6*6,6))
    qint_error=np.zeros((6,1))

    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    time.sleep(1)

    arm.reset(wait=True)

    arm.set_mode(xarm6_mode)
    arm.set_state(0)
    time.sleep(1)
    
    ### Define for trajectory
    traj = Trajectory(6)
    traj_flag = [0]*6
    motion = 1
    target_q = [0]*6
    
    #"""
    ### xarm6 Kinematic parameters based on URDF file for MR
    link_01 = np.array([0, 0, 0.267]); link_12 = np.array([0, 0, 0]); link_23 = np.array([0.0535, 0, 0.2845]);
    link_34 = np.array([0.0775, 0, -0.3425]); link_45 = np.array([0, 0, 0]); link_56 = np.array([0.076, 0, -0.097]); link_6E = np.array([0.0, 0, 0.0])
    w=np.array([[0,0,1],[0,1,0],[0,1,0],[0,0,-1],[0,1,0],[0,0,-1]])
    L_=np.array([(link_01+link_12)[:], (link_01+link_12+link_23)[:],(link_01+link_12+link_23+link_34)[:],(link_01+link_12+link_23+link_34+link_45)[:]\
    ,(link_01+link_12+link_23+link_34+link_45+link_56)[:],(link_01+link_12+link_23+link_34+link_45+link_56+link_6E)[:]])
    P_=np.array([link_01[:],(link_01+link_12)[:], (link_01+link_12+link_23)[:],(link_01+link_12+link_23+link_34)[:],(link_01+link_12+link_23+link_34+link_45)[:]\
    ,(link_01+link_12+link_23+link_34+link_45+link_56)[:]])
    v = np.array([-mr.VecToso3(w[0,:])@P_[0,:],-mr.VecToso3(w[1,:])@P_[1,:],-mr.VecToso3(w[2,:])@P_[2,:],-mr.VecToso3(w[3,:])@P_[3,:],-mr.VecToso3(w[4,:])@P_[4,:],-mr.VecToso3(w[5,:])@P_[5,:]])
    Slist = np.transpose(np.array([np.append(w[0,:],v[0,:]),np.append(w[1,:],v[1,:]),np.append(w[2,:],v[2,:]),np.append(w[3,:],v[3,:]),np.append(w[4,:],v[4,:]),np.append(w[5,:],v[5,:])]))
    print(np.transpose(Slist))
    T_0 = np.array([[1,0,0,L_[-1,0]],[0,-1,0,L_[-1,1]],[0,0,-1,L_[-1,2]],[0,0,0,1]])
    print(T_0)
    #T_0[0:3,0:3] = tf.transformations.euler_matrix(0,-100*deg2rad,0)[0:3,0:3]
    Blist = mr.Adjoint(mr.TransInv(T_0))@Slist
    J_b_mr = np.zeros((6,6))
    J_s_mr = np.zeros((6,6))

    M_a_ = np.diag([2,2,2,0.5,0.5,0.25])
    D_a_ = np.diag([15,15,15,5,5,5])

    V_d = np.array([0,0,0,0,0,0])
    X_d = np.array([350,361,280,45,-10,0])
    
    frq_cutoff = 50
    alpha = (frq_cutoff*(1/xarm6_frq))/(1+frq_cutoff*(1/xarm6_frq))
    F_ext_buf = np.array([0]*6)
    F_ext_off = np.array([0]*6)    
    #"""

    readdata=arm.get_joint_states()[1]
    init_pos = [readdata[0][0],readdata[0][1],readdata[0][2],readdata[0][3],readdata[0][4],readdata[0][5]]
    print(init_pos)

    
    
    ### realtime loop start
    while arm.connected:
        g_time = time.time()
        readdata=arm.get_joint_states()[1]

        q = [readdata[0][0],readdata[0][1],readdata[0][2],readdata[0][3],readdata[0][4],readdata[0][5]]
        q_dot = [readdata[1][0],readdata[1][1],readdata[1][2],readdata[1][3],readdata[1][4],readdata[1][5]]
        q_rad = [readdata[0][0]*deg2rad,readdata[0][1]*deg2rad,readdata[0][2]*deg2rad,readdata[0][3]*deg2rad,readdata[0][4]*deg2rad,readdata[0][5]*deg2rad]
        q_dot_rad = [readdata[1][0]*deg2rad,readdata[1][1]*deg2rad,readdata[1][2]*deg2rad,readdata[1][3]*deg2rad,readdata[1][4]*deg2rad,readdata[1][5]*deg2rad]
        

        #"""
        T=T_0
        for i in range(len(q)-1,-1,-1):
            T=np.dot(mr.MatrixExp6(mr.VecTose3(Slist[:,i]*np.array(q_rad[i]))),T)

        J_b_mr[0:3,:] = mr.JacobianBody(Blist,q_rad)[3:6]
        J_b_mr[3:6,:] = mr.JacobianBody(Blist,q_rad)[0:3]
        #J_s_mr[0:3,:] = mr.JacobianSpace(Slist,q_rad)[3:6]
        #J_s_mr[3:6,:] = mr.JacobianSpace(Slist,q_rad)[0:3]

        #for i in range(0,6):
        #    for j in range(0,6):
        #        Js_dot[6*i:6*i+6,j] = mr.ad(J_s_mr[:,i])@J_s_mr[:,j]

        #print(T)
        F_=np.array([[5,0,0,0,0,0],[0,5,0,0,0,0],[0,0,5,0,0,0],[0,0,0,25,0,0],[0,0,0,0,25,0],[0,0,0,0,0,25]])
        F_ext_tmp = np.transpose(np.dot(F_,np.transpose(np.array([_F['force']+_F['torque']]))))
        F_ext = alpha*(F_ext_tmp-F_ext_off)+(1-alpha)*F_ext_buf
        F_ext_buf = F_ext
        tau_ext = np.transpose(J_b_mr)@np.transpose(F_ext)
        print(tau_ext)
        #"""
        if motion==1 and traj_flag[-1]==0:
            #target_q=[0.0*deg2rad, 0.0*deg2rad, 0.0*deg2rad, 0.0*deg2rad, 0.0*deg2rad, 0.0*deg2rad]
            target_q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            motion+=1
            traj_flag=[1]*6
        elif motion==2 and traj_flag[-1]==0:
            #target_q=[-10.0*deg2rad, -10.0*deg2rad, 20.0*deg2rad, -30.0*deg2rad, 10.0*deg2rad, 20.0*deg2rad]
            target_q=[0.0, -60.0, -30.0, 0.0, 90.0, 0.0]
            motion+=1
            traj_flag=[1]*6
        elif motion==3 and traj_flag[-1]==0:
            #target_q=[-25.0*deg2rad, 0.0*deg2rad, 10.0*deg2rad, -50.0*deg2rad, 20.0*deg2rad, 40.0*deg2rad]
            target_q=[-90.0, -60.0, -30.0, 0.0, 90.0, 0.0]
            motion+=1
            traj_flag=[1]*6
        elif motion==4 and traj_flag[-1]==0:
            #target_q=[-50.0*deg2rad, 50.0*deg2rad, 50.0*deg2rad, 50.0*deg2rad, 50.0*deg2rad, 50.0*deg2rad]
            target_q=[0.0, -60.0, -30.0, 0.0, 90.0, 0.0]
            motion+=1
            traj_flag=[1]*6
        elif motion==5 and traj_flag[-1]==0:
            #target_q=[-30.0*deg2rad, 10.0*deg2rad, 30.0*deg2rad, -20.0*deg2rad, 10.0*deg2rad, 60.0*deg2rad]
            target_q=[0.0, 90.0, -150.0, 80.0, -80.0, 0.0]
            motion+=1
            traj_flag=[1]*6
        elif motion==6 and traj_flag[-1]==0:
            #target_q=[-20.0*deg2rad, 20.0*deg2rad, 40.0*deg2rad, 20.0*deg2rad, 0.0*deg2rad, 90.0*deg2rad]
            target_q=[90.0, -60.0, -30.0, -80.0, 90.0, 0.0]
            motion=1
            traj_flag=[1]*6
        
            
        for i in range(6):
            if traj_flag[i]==1:
                traj.SetPolynomial5th(i,q[i],target_q[i],g_time,3.0)
                qd[i]=q[i]
                traj_flag[i]=2

            elif traj_flag[i]==2:
                tmp_res,tmp_flag=traj.Polynomial5th(i,g_time)
                qd[i] = tmp_res[0]
                qd_dot[i] = tmp_res[1]
                qd_ddot[i] = tmp_res[2]

                if tmp_flag == 0:
                    if motion == 3:
                        traj_flag[i]=-1
                    else:
                        traj_flag[i]=0

        for i in range(6):
            if i == 0:
                cmd_vel[i] = 10*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])+1*tau_ext[i]
                #cmd_vel[i] = 0.65*tau_ext[i]
            elif i == 1:
                cmd_vel[i] = 10*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])+1*tau_ext[i]
                #cmd_vel[i] = 0.65*tau_ext[i]
            elif i == 2:
                cmd_vel[i] = 10*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])+1*tau_ext[i]
                #cmd_vel[i] = 0.65*tau_ext[i]
            elif i == 3:
                cmd_vel[i] = 10*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])+1*tau_ext[i]
                #cmd_vel[i] = 0.65*tau_ext[i]
            elif i == 4:
                cmd_vel[i] = 10*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])+1*tau_ext[i]
                #cmd_vel[i] = 0.65*tau_ext[i]
            elif i == 5:
                cmd_vel[i] = 10*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])+1*tau_ext[i]
                #cmd_vel[i] = 0.65*tau_ext[i]
        #print(qd)
        #print(q)
        #print(cmd_vel)
        
        if xarm6_mode == 4:
            arm.vc_set_joint_velocity(cmd_vel)

        
        while time.time()-g_time<(1/xarm6_frq):
            time.sleep(1/1000000)
            glob_time_buf=time.time()
            init_time_buf=init_time
        print(time.time()-g_time)
        
if __name__=='__main__':

    xarm6_cmd_task = Process(target=xarm6_cmd_run, args=())
    ft_sensor_task = Process(target=ft_sensor_run, args=())
   
    try:
        xarm6_cmd_task.start()
        ft_sensor_task.start()

        xarm6_cmd_task.join()
        ft_sensor_task.join()

    except KeyboardInterrupt:
        ch = setUpChannel(channel=0)
        frame = Frame(
            id_=0x64,
            data=[0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            dlc=8
        )
        ch.write(frame)
        xarm6_cmd_task.terminate()
        ft_sensor_task.terminate()
