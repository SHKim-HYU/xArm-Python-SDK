#!/usr/bin/env python3
from math import sin, cos, tan, pi
from trajectory_generate import Trajectory

import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import modern_robotics as mr
from tasho import robot as rob
from tasho import world_simulator
import pybullet as p

import casadi as cs
import tf

from multiprocessing import Process, Manager


sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

Simulation = True

n_dof = 6
xarm6_mode = 2 #4 #                               2: teaching mode, 4: joint velocity control
control_mode = 0 #1 #2 #3                         0: PI velocity, 1: Gravity, 2: IDC 3: FDCC
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



def xarm6_cmd_run():
    deg2rad = 3.141592/180
    rad2deg = 180/3.141592
    xarm6_frq = 250
    
    robot_choice = 'xarm6'
    robot = rob.Robot(robot_choice)
    
    jac_fun = robot.set_kinematic_jacobian(name="jac_fun",q=6,frame='space')


    
    init_time=time.time()
    
    qd=np.array([0.0]*n_dof)
    qd_dot=np.array([0.0]*n_dof)
    qd_ddot=np.array([0.0]*n_dof)
    cmd_vel=np.array([0.0]*n_dof)
    
    Js_dot = np.zeros((6*6,6))
    qint_error=np.zeros((n_dof,1))

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
    traj = Trajectory(n_dof)
    traj_flag = [0]*n_dof
    motion = 1
    target_q = np.array([0.0]*n_dof)
    
    #"""
    ### xarm6 Kinematic parameters based on URDF file for MR
    link_01 = np.array([0, 0, 0.267]); link_12 = np.array([0, 0, 0]); link_23 = np.array([0.0535, 0, 0.2845]);
    link_34 = np.array([0.0775, 0, -0.3425]); link_45 = np.array([0, 0, 0]); link_56 = np.array([0.076, 0, -0.097]); link_6E = np.array([0.0, 0, -0.0045])
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
    
       
    tau_buf = np.array([0.0]*6)
    #"""

    readdata=arm.get_joint_states()[1]
    init_pos = [readdata[0][0],readdata[0][1],readdata[0][2],readdata[0][3],readdata[0][4],readdata[0][5]]
    print(init_pos)
    q_buf=np.array([0.0]*6)

    
    
    ### realtime loop start
    while arm.connected:
        g_time = time.time()
        readdata=arm.get_joint_states()[1]

        #q = [readdata[0][0],readdata[0][1],readdata[0][2],readdata[0][3],readdata[0][4],readdata[0][5]]
        #q_dot = [readdata[1][0],readdata[1][1],readdata[1][2],readdata[1][3],readdata[1][4],readdata[1][5]]
        q = np.array([readdata[0][0]*deg2rad,readdata[0][1]*deg2rad,readdata[0][2]*deg2rad,readdata[0][3]*deg2rad,readdata[0][4]*deg2rad,readdata[0][5]*deg2rad])
        q_dot = np.array([readdata[1][0]*deg2rad,readdata[1][1]*deg2rad,readdata[1][2]*deg2rad,readdata[1][3]*deg2rad,readdata[1][4]*deg2rad,readdata[1][5]*deg2rad])
        tau = np.array(arm.joints_torque[0:6])
        #print(q)
        #print(tau)
        T=np.array(robot.fk(q)[n_dof])
        
        #print(tf.transformations.euler_from_matrix(T))
        """
        T=T_0
        for i in range(len(q)-1,-1,-1):
            T=np.dot(mr.MatrixExp6(mr.VecTose3(Slist[:,i]*np.array(q[i]))),T)
        """
        
        J_s = robot.J_s(q)
        J_b = robot.J_b(q)
        
        #print(J_s)
        print(J_b)
        
        
        #for i in range(0,6):
        #    for j in range(0,6):
        #        Js_dot[6*i:6*i+6,j] = mr.ad(J_s_mr[:,i])@J_s_mr[:,j]
        
        
        M_=robot.M(q).full().reshape(len(q),len(q))
        C_=robot.C(q,q_dot).full().reshape(len(q),len(q))
        G_=robot.G(q).full().reshape(len(q))
        
        #print(M_)
        #print(C_)
        #print(T)

        F_ext = np.array([_F['force']+_F['torque']])
        tau_ext = np.transpose(J_b_mr)@np.transpose(F_ext)
        #print(tau_ext)
        
        #"""
        if motion==1 and traj_flag[-1]==0:
            #target_q=np.array([0.0*deg2rad, 0.0*deg2rad, 0.0*deg2rad, 0.0*deg2rad, 0.0*deg2rad, 0.0*deg2rad])
            target_q=np.array([0.0*deg2rad, -60.0*deg2rad, -30.0*deg2rad, 0.0*deg2rad, 90.0*deg2rad, 0.0*deg2rad])
            motion+=1
            traj_flag=[1]*6
        elif motion==2 and traj_flag[-1]==0:
            target_q=np.array([0.0*deg2rad, -60.0*deg2rad, -30.0*deg2rad, 0.0*deg2rad, 90.0*deg2rad, 0.0*deg2rad])
            motion+=1
            traj_flag=[1]*6
        elif motion==3 and traj_flag[-1]==0:
            target_q=np.array([-90.0*deg2rad, -60.0*deg2rad, -30.0*deg2rad, 0.0*deg2rad, 90.0*deg2rad, 0.0*deg2rad])
            motion+=1
            traj_flag=[1]*6
        elif motion==4 and traj_flag[-1]==0:
            target_q=np.array([0.0*deg2rad, -60.0*deg2rad, -30.0*deg2rad, 0.0*deg2rad, 90.0*deg2rad, 0.0*deg2rad])
            motion+=1
            traj_flag=[1]*6
        elif motion==5 and traj_flag[-1]==0:
            target_q=np.array([0.0*deg2rad, 90.0*deg2rad, -150.0*deg2rad, 80.0*deg2rad, -80.0*deg2rad, 0.0*deg2rad])
            motion+=1
            traj_flag=[1]*6
        elif motion==6 and traj_flag[-1]==0:
            target_q=np.array([90.0*deg2rad, -60.0*deg2rad, -30.0*deg2rad, -80.0*deg2rad, 90.0*deg2rad, 0.0*deg2rad])
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
                    if motion == 2:
                        traj_flag[i]=0
                    else:
                        traj_flag[i]=0            
        if control_mode == 0: #PI-velocity
            cmd_vel = np.dot(np.diag([8,8,8,5,5,5]),qd-q)+np.dot(np.diag([0.08,0.08,0.08,0.05,0.05,0.05]),qd_dot-q_dot)+np.transpose(tau_ext)[0]

        elif control_mode == 1: #Gravity
            print("gravity= ",G_)
            print("position= ",q)
            print("velocity= ",q_dot)
            print("actual= ",tau)
            print('\n\n')
            
            cmd_vel = G_-tau+np.transpose(tau_ext)[0]
            #cmd_vel = np.transpose(tau_ext)[0]
            
        elif control_mode == 2: #IDC based
            """
            q_buf=q_buf+1/xarm6_frq*(np.array(qd)-np.array(q))
            u_0 = np.dot(np.diag([8,8,8,5,5,5]),(np.array(qd)-np.array(q)))+np.dot(np.diag([0.08,0.08,0.08,0.05,0.05,0.05]),(np.array(qd_dot)-np.array(q_dot)))\
                  +np.dot(np.diag([20,20,20,10,10,10]),q_buf)
            tau_d = np.dot(M_,u_0)+np.dot(C_,q_dot)+np.array(G_)#+np.transpose(tau_ext)[0]
            tau_buf=tau_buf+1/xarm6_frq*(tau_d-tau)
            cmd_vel = np.dot(np.diag([10,10,10,2,2,2]),(tau_d-tau))+np.dot(np.diag([100,100,100,20,20,20]),(tau_buf))
            print(cmd_vel)
            """
            q_buf=q_buf+1/xarm6_frq*(np.array(qd)-np.array(q))
            u_0 = np.dot(np.diag([12,12,12,7.5,10,7.5]),(np.array(qd)-np.array(q)))+np.dot(np.diag([0.12,0.12,0.12,0.075,0.01,0.075]),(np.array(qd_dot)-np.array(q_dot)))#\
                  #+np.dot(np.diag([30,30,30,15,15,15]),q_buf)
            tau_d = np.dot(M_,u_0)+np.dot(C_,q_dot)#+np.transpose(tau_ext)[0]#+np.array(G_)#
            tau_buf=tau_buf+1/xarm6_frq*(tau_d)
            #cmd_vel = (0.03*(tau_d-tau)+0.3*(tau_buf))*rad2deg
            cmd_vel = np.dot(np.diag([1,1,1,0.2,0.7,0.2]),(tau_d))#+np.dot(np.diag([10,10,10,2,7,2]),(tau_buf))
            #tau_buf = tau_d-tau
            #q_buf = np.array(qd)-np.array(q))
            print(cmd_vel)
            #"""
        
        elif control_mode == 3: # FDCC
           
           
           
           pass
        
        if xarm6_mode == 4:
            arm.vc_set_joint_velocity(cmd_vel*rad2deg)
            #print(cmd_vel)
            pass

        
        while time.time()-g_time<(1/xarm6_frq):
            time.sleep(1/1000000)
            glob_time_buf=time.time()
            init_time_buf=init_time
        print(time.time()-g_time)
        
if __name__=='__main__':

    xarm6_cmd_task = Process(target=xarm6_cmd_run, args=())
    
    try:
        xarm6_cmd_task.start()
       
        xarm6_cmd_task.join()
        
    except KeyboardInterrupt:

        xarm6_cmd_task.terminate()
        print("job done")
