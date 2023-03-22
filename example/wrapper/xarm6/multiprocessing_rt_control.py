#!/usr/bin/env python3
from math import sin, cos, tan, pi
from trajectory_generate import Trajectory

import time
import os
import sys
import numpy as np
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

_q['q'] = [0.0]*9; _q['q_dot'] = [0.0]*9; _q['trq_g'] = [0.0]*9; _q['mpc_q']=[0.0]*9; _q['mpc_q_dot']=[0.0]*9;
_q['trq_ext'] = [0.0]*6;
_q_d['qd'] = [0.0]*9; _q_d['qd_dot'] = [0.0]*9; _q_d['qd_ddot'] = [0.0]*9;



def xarm6_cmd_run():
    deg2rad = 3.141592/180
    rad2deg = 180/3.141592
    xarm6_frq = 1000

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

    # set joint velocity control mode
    arm.set_mode(4)
    arm.set_state(0)
    time.sleep(1)
    
    ### Define for trajectory
    traj = Trajectory(6)
    traj_flag = [0]*6
    motion = 1
    target_q = [0]*6
    
    """
    ### xarm6 Kinematic parameters based on URDF file for MR
    link_01 = np.array([0, 0, 0.135]); link_12 = np.array([0, -0.0062, 0]); link_23 = np.array([0, 0, 0.411]);
    link_34 = np.array([0, 0, 0.368]); link_45 = np.array([0, 0, 0]); link_56 = np.array([0, 0, 0.121]); link_6E = np.array([-0.012, 0, 0.2205])
    w=np.array([[0,0,1],[0,1,0],[0,1,0],[0,0,1],[0,1,0],[0,0,1]])
    L_=np.array([(link_01+link_12)[:], (link_01+link_12+link_23)[:],(link_01+link_12+link_23+link_34)[:],(link_01+link_12+link_23+link_34+link_45)[:]\
    ,(link_01+link_12+link_23+link_34+link_45+link_56)[:],(link_01+link_12+link_23+link_34+link_45+link_56+link_6E)[:]])
    P_=np.array([link_01[:],(link_01+link_12)[:], (link_01+link_12+link_23)[:],(link_01+link_12+link_23+link_34)[:],(link_01+link_12+link_23+link_34+link_45)[:]\
    ,(link_01+link_12+link_23+link_34+link_45+link_56)[:]])
    v = np.array([-mr.VecToso3(w[0,:])@P_[0,:],-mr.VecToso3(w[1,:])@P_[1,:],-mr.VecToso3(w[2,:])@P_[2,:],-mr.VecToso3(w[3,:])@P_[3,:],-mr.VecToso3(w[4,:])@P_[4,:],-mr.VecToso3(w[5,:])@P_[5,:]])
    Slist = np.transpose(np.array([np.append(w[0,:],v[0,:]),np.append(w[1,:],v[1,:]),np.append(w[2,:],v[2,:]),np.append(w[3,:],v[3,:]),np.append(w[4,:],v[4,:]),np.append(w[5,:],v[5,:])]))
    T_0 = np.array([[1,0,0,L_[-1,0]],[0,1,0,L_[-1,1]],[0,0,1,L_[-1,2]],[0,0,0,1]])
    T_0[0:3,0:3] = tf.transformations.euler_matrix(0,-100*deg2rad,0)[0:3,0:3]
    Blist = mr.Adjoint(mr.TransInv(T_0))@Slist
    J_b_mr = np.zeros((6,6))
    J_s_mr = np.zeros((6,6))

    M_a_ = np.diag([2,2,2,0.5,0.5,0.25])
    D_a_ = np.diag([15,15,15,5,5,5])

    V_d = np.array([0,0,0,0,0,0])
    X_d = np.array([350,361,280,45,-10,0])
    """

    readdata=arm.get_joint_states()[1]
    init_pos = [readdata[0][0],readdata[0][1],readdata[0][2],readdata[0][3],readdata[0][4],readdata[0][5]]
    print(init_pos)

    
    
    ### realtime loop start
    while arm.connected:
        g_time = time.time()
        readdata=arm.get_joint_states()[1]

        q = [readdata[0][0],readdata[0][1],readdata[0][2],readdata[0][3],readdata[0][4],readdata[0][5]]
        q_dot = [readdata[1][0],readdata[1][1],readdata[1][2],readdata[1][3],readdata[1][4],readdata[1][5]]
        
        """
        pose_tcp = [readdata.data.actual_tcp_position[0],readdata.data.actual_tcp_position[1],readdata.data.actual_tcp_position[2],readdata.data.actual_tcp_position[3],readdata.data.actual_tcp_position[4],readdata.data.actual_tcp_position[5]]
        T = tf.transformations.euler_matrix(pose_tcp[3],pose_tcp[4],pose_tcp[5])
        T[0:3,3] = pose_tcp[0:3]

        J_b_mr[0:3,:] = mr.JacobianBody(Blist,q)[3:6]
        J_b_mr[3:6,:] = mr.JacobianBody(Blist,q)[0:3]
        J_s_mr[0:3,:] = mr.JacobianSpace(Slist,q)[3:6]
        J_s_mr[3:6,:] = mr.JacobianSpace(Slist,q)[0:3]
        J_s= np.array([readdata.data.jacobian_matrix[0].data, readdata.data.jacobian_matrix[1].data, readdata.data.jacobian_matrix[2].data, readdata.data.jacobian_matrix[3].data, readdata.data.jacobian_matrix[4].data, readdata.data.jacobian_matrix[5].data])
        J_b= mr.Adjoint(mr.TransInv(mr.FKinSpace(T_0,Slist,q)))@J_s
        for i in range(0,6):
            for j in range(0,6):
                Js_dot[6*i:6*i+6,j] = mr.ad(J_s[:,i])@J_s[:,j]


        # print(Jb_dot)
        
        M_=np.array([readdata.data.mass_matrix[0].data, readdata.data.mass_matrix[1].data, readdata.data.mass_matrix[2].data, readdata.data.mass_matrix[3].data, readdata.data.mass_matrix[4].data, readdata.data.mass_matrix[5].data])

        C_=np.array([readdata.data.coriolis_matrix[0].data, readdata.data.coriolis_matrix[1].data, readdata.data.coriolis_matrix[2].data, readdata.data.coriolis_matrix[3].data, readdata.data.coriolis_matrix[4].data, readdata.data.coriolis_matrix[5].data])

        G_=np.array(tor_g)

        print(tor_g)
        """
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
                traj.SetPolynomial5th(i,q[i],target_q[i],g_time,5.0)
                qd[i]=q[i]
                traj_flag[i]=2

            elif traj_flag[i]==2:
                tmp_res,tmp_flag=traj.Polynomial5th(i,g_time)
                qd[i] = tmp_res[0]
                qd_dot[i] = tmp_res[1]
                qd_ddot[i] = tmp_res[2]

                if tmp_flag == 0:
                    traj_flag[i]=0

        for i in range(6):
            if i == 0:
                cmd_vel[i] = 5*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])
            elif i == 1:
                cmd_vel[i] = 5*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])
            elif i == 2:
                cmd_vel[i] = 5*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])
            elif i == 3:
                cmd_vel[i] = 5*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])
            elif i == 4:
                cmd_vel[i] = 5*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])
            elif i == 5:
                cmd_vel[i] = 5*(qd[i]-q[i])+0.1*(qd_dot[i]-q_dot[i])
        #print(qd)
        print(q)
        #print(cmd_vel)
        arm.vc_set_joint_velocity(cmd_vel)

        
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
