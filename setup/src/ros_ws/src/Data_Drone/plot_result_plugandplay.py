import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.join(current_dir,'..')
sys.path.append(project_folder)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set print options for numpy
np.set_printoptions(precision=5, suppress=True)

# from trajectory_generation.generate_traj import *
def rms(X):
    y = np.sqrt(np.mean(X ** 2))
    return y

URI0_ = 'radio://0/80/2M/E7E7E7E7E1'
URI0 = 'radio://0/80/2M/E7E7E7E7E5'
URI1 = 'radio://0/80/2M/E7E7E7E7E7'
URI2 ='radio://0/80/2M/E7E7E7E7E8'
URI2_ ='radio://0/80/2M/E7E7E7E7E2'
URI3 ='radio://0/80/2M/E7E7E7E7E9'
URI4 ='radio://0/80/2M/E7E7E7E7E3'

URI_s1 = 'udp://0.0.0.0:19850'
URI_s2 = 'udp://0.0.0.0:19851'
URI_s3 = 'udp://0.0.0.0:19852'
URI_s4 = 'udp://0.0.0.0:19853'


def load_data(path):
    d = np.load(path, allow_pickle=True).item()
    return d['data'], d['ref'], d['vref']

def quaternion_to_euler(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Clamping to 90 degrees
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]

def plot_positions(t, state_xi, pos_ref, takeoff_line=None, id=0, title=None):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['x (m)', 'y (m)', 'z (m)']
    for i in range(3):
        axs[i].plot(t, state_xi[:, i], label='sim')
        axs[i].plot(t, pos_ref[:, i], '--', label='ref')
        axs[i].legend(loc="best")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        
    if takeoff_line is not None:
        for ax in axs:
            ax.axvline(takeoff_line, color='g', linestyle='--', alpha=0.7)

    axs[2].set_xlabel('Time (s)')

    if title is None:
        fig.suptitle(f'Drone {id+1} Positions', fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)

def pose_with_ref_velocity(t_pose, state_xi, t_ref, pos_ref, id=0, title=None):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['vx (m/s)', 'vy (m/s)', 'vz (m/s)']
    for i in range(3):
        axs[i].plot(t_pose, state_xi[:, i], label='sim')
        axs[i].plot(t_ref, pos_ref[:, i], '--', label='ref')
        axs[i].legend(loc="best")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')

    if title is None:
        fig.suptitle(f'Drone {id+1} Velocities', fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)


def plot_velocity(t, state_xi, pos_ref, takeoff_line=None, id=0, title=None):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['vx (m/s)', 'vy (m/s)', 'vz (m/s)']
    for i in range(3):
        axs[i].plot(t, state_xi[:, i], label='sim')
        axs[i].plot(t, pos_ref[:, i], '--', label='ref')
        axs[i].legend(loc="best")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        
    if takeoff_line is not None:
        for ax in axs:
            ax.axvline(takeoff_line, color='g', linestyle='--', alpha=0.7)

    axs[2].set_xlabel('Time (s)')

    if title is None:
        fig.suptitle(f'Drone {id+1} Velocities', fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)


def plot_v(t, v, vref, id=0, title=None):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['ax (m^2/s)', 'ay (m^2/s)', 'az (m^2/s)']
    for i in range(3):
        axs[i].plot(t, v[:, i], label='sim')
        axs[i].plot(t, vref[:, i], '--', label='ref')
        axs[i].legend(loc="best")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')

    if title is None:
        fig.suptitle(f'Drone {id+1} Accelerations', fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)


def plot_pose_velocity(t, state_xi, id=0, title=None):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['vx (m/s)', 'vy (m/s)', 'vz (m/s)']
    for i in range(3):
        axs[i].plot(t, state_xi[:, i], label='sim')
        axs[i].legend(loc="best")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')

    if title is None:
        fig.suptitle(f'Drone {id+1} Pose Velocities at 100 Hz', fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)


def plot_angles(t, state_eta, controls, eps_max, takeoff_line=None, id=0, title=None, option='rad'):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    if option == 'rad':
        labels = [r'$\phi$ (rad)', r'$\theta$ (rad)', r'$\psi$ (rad)']
    else:
        labels = [r'$\phi$ (deg)', r'$\theta$ (deg)', r'$\psi$ (deg)']
        state_eta = state_eta * 180 / np.pi
        controls[:, 1:2] = controls[:, 1:2] * 180 / np.pi

    for i in range(3):
        axs[i].plot(t, state_eta[:, i], label='sim')
        if i != 2:
            axs[i].plot(t, controls[:, i+1], '--', label='ref')
            axs[i].axhline(eps_max, color='r', linestyle='--', alpha=0.7)
            axs[i].axhline(-eps_max, color='r', linestyle='--', alpha=0.7)

        axs[i].legend(loc="best")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        
    if takeoff_line is not None:
        for ax in axs:
            ax.axvline(takeoff_line, color='g', linestyle='--', alpha=0.7)

    axs[2].set_xlabel('Time (s)')

    if title is None:
        fig.suptitle(f'Drone {id+1} Angles', fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)


def plot_position_difference(t, pose_data, qtm_data, id=0):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['ex (m)', 'ey (m)', 'ez (m)']
    for i in range(3):
        axs[i].plot(t, pose_data[:, i] - qtm_data[:, i], label='sim')
        axs[i].legend(loc="best")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')
    fig.suptitle(f'Drone {id+1} Position Difference POSE - QTM', fontsize=14)


def plot_velocity_difference(t, pose_data, qtm_data, id=0):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['evx (m/s)', 'evy (m/s)', 'evz (m/s)']
    for i in range(3):
        axs[i].plot(t, pose_data[:, i] - qtm_data[:, i], label='sim')
        axs[i].legend()
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')
    fig.suptitle(f'Drone {id+1} Velocity Difference POSE - QTM', fontsize=14)


def plot_controls(t, angles, controls, takeoff_line=None, id=0, Tmax=None, eps_max=None, title=None):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['Thrust', 'Roll (rad)', 'Pitch (rad)']
    axs[0].plot(t, controls[:, 0], label='thrust')
    axs[1].plot(t, controls[:, 1], label='desired roll')
    axs[2].plot(t, controls[:, 2], label='desired pitch')
    for i in range(3):
        # axs[i].plot(t, controls[:, i], label='ref')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    if Tmax is not None:
        axs[0].axhline(Tmax, color='r', linestyle='--', alpha=0.7)
        axs[0].axhline(0, color='r', linestyle='--', alpha=0.7)

    if eps_max is not None:
        axs[1].plot(t, angles[:, 0], '--', label='measured roll')
        axs[2].plot(t, angles[:, 1], '--', label='measured pitch')
        for i in [1, 2]:
            axs[i].axhline(eps_max, color='r', linestyle='--', alpha=0.7)
            axs[i].axhline(-eps_max, color='r', linestyle='--', alpha=0.7)
            
    if takeoff_line is not None:
        for ax in axs:
            ax.axvline(takeoff_line, color='g', linestyle='--', alpha=0.7)

    for ax in axs:
        ax.legend(loc='upper right')
    axs[2].set_xlabel('Time (s)')

    if title is None:
        fig.suptitle(f'Drone {id+1} Controls', fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)


def plot_yaw(t, yaws, id=0, title=None, option='rad'):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if option == 'deg':
        ax.plot(t, yaws * 180/np.pi, label=r'$\psi$')
        ax.set_ylabel('Yaw (deg)')
    elif option == 'rad':
        ax.plot(t, yaws, label=r'$\psi$')
        ax.set_ylabel('Yaw (rad)')

    ax.set_xlabel('Time (s)')
    ax.grid(True)
    ax.legend()

    if title is None:
        fig.suptitle(f'Drone {id+1} Measured Yaw', fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)

if __name__=='__main__':
    
    # sim_id = 1707240021
    # sim_id = 1907240013
    # sim_id = 19072403
    # sim_id = 60320250001
    # sim_id = 70320250002
    # sim_id = 100320250006
    # sim_id = 110320250024
    # sim_id = 140320250001
    sim_id = 9

    # list_of_bodies = ['DroneE7', 'DroneE8','DroneE9']
    # uris = [URI1,URI2,URI3]
    # list_of_bodies = ['DroneE1', 'DroneE2','DroneE3','DroneE9']
    # uris = [URI0_,URI2_,URI4,URI3]
    # list_of_bodies = ['DroneE1','DroneE9' ,'DroneE3','DroneE2']
    # uris = [URI0_,URI3,URI4,URI2_]
    # list_of_bodies = ['DroneE9','DroneE2']
    # uris = [URI3,URI2_]
    # list_of_bodies = ['DroneE1']
    # list_of_bodies = ['crazyflie_1','crazyflie_2','crazyflie_3','crazyflie_4']
    list_of_bodies = ['DroneE1','DroneE4','DroneE7','DroneE9']
    uris = ['radio://0/80/2M/E7E7E7E7E1',
            'radio://0/80/2M/E7E7E7E7E4',
            'radio://0/80/2M/E7E7E7E7E7',
            'radio://0/80/2M/E7E7E7E7E9']

    # uris = [URI0_]
    Ts = 0.1
    Tsim = 30
    # time_stamp=np.arange(start=0,step=Ts,stop=Tsim)
    # print(np.size(time_stamp,0))
    # print('Time stamp : ',len(time_stamp))
    # load_data = np.load('./Data/PX_data_drone_takeoff_{}.npy'.format(sim_id),allow_pickle=True).item()
    # load_data = np.load('./Data/PX_data_drone_landing_{}.npy'.format(sim_id),allow_pickle=True).item()

    simu = np.load('./rds_sim_{}.npy'.format(sim_id),allow_pickle=True).item()

    # generate reference trajectory for leader
    # Xi=[[0, 0, 0, -1.2187, -1.4288, -0.8787, 1.4708, 0.5615, 0, 0, 0],
    #     [0, 0, 0, 0.3802, 0.5887, 0.5888, 0.2743, -0.9708, 0, 0, 0],
    #     [0, 0, 0, 0.2649, 0.2032, 0.9542, 0.7308, 0.5755, 0, 0, 0]]
    
    Tf = 30     # total simulation time
    # ref = generate_trajectory_Bspline(Xi, [0,Tf], 4, 0.1)
    # ref_original = generate_trajectory_Bspline(Xi, [0,Tf], 4, 0.1)
    # ref_l = shift_trajectory(ref_original, 0, 0, 0.8)
    
    # Nb = len(ref['time_step'])-1

    # replicator dynamics design
    offset_pos = np.array([2.01, 2.01, 2.01]).reshape(-1, 1)
    # ref_rd = ref['trajectory'][0:3,:] + offset_pos

    # ptf = "./Trajectory/gap_ctrl_pts_Vincent.mat"


    # gap, gap_vel, gap_acc = generate_gaps_4agents_Vincent(ptf=ptf,Tf=Tf,h=0.1,timing=[0.2,0.4,0.2,0.2])
    # gap, gap_vel, gap_acc = generate_gaps_4agents_Vincent_corrected(ptf=ptf,Tf=Tf,h=0.1,timing=[0.21,0.22,0.28,0.29])
    # gap, gap_vel, gap_acc = generate_gaps_4agents_Vincent_corrected(ptf=ptf,Tf=Tf,h=0.1,timing=[0.28,0.16,0.28,0.28])
    # gap, gap_vel, gap_acc = generate_gaps_4agents_Vincent_corrected(ptf=ptf,Tf=Tf,h=0.1,timing=[0.2,0.2,0.3,0.3])

    ## Generate takeoff and landing trajectories ##
    # ref_takeoff, ref_landing = generate_takeoff_landing_traj(ref_l, gap, T_takeoff=7, T_hovering_1=1, T_hovering_2=1, T_landing=5, Ts=0.1)

    # ref_l, gap, gap_vel, gap_acc = get_full_traj_4agents_game(ref_l, gap, gap_vel, gap_acc, ref_takeoff, ref_landing)


    # load_data = np.load('./Data/PX_data_drone_{}.npy'.format(sim_id),allow_pickle=True).item()

    fl_data, fl_ref, fl_vref = load_data('./drone_data_{}_seq.npy'.format(sim_id))
    nb = len(fl_data)
    drone_names = fl_data.keys()
    all_names = list(fl_data.keys())
    pos_dict = {}
    ref_dict = {}
    t_dicts = []

    for name in drone_names:
        # Get the normal data after takeoff, when refrence tracking
        track_data = fl_data[name]

        track_state = track_data[:, 0:6]
        track_angles = np.array([quaternion_to_euler(track_data[i, 6:10]) for i in range(len(track_data))])
        track_controls = track_data[:, 13:16]
        track_yaws = track_data[:, 16]
        track_t = track_data[:, 17].reshape(-1, 1)
        track_v = track_data[:, 18:21]
        track_ref = track_data[:, 21:27]
        track_vref = track_data[:, 27:30]

        plot_positions(track_t, track_state[:, 0:3], track_ref[:, 0:3], takeoff_line=None, id=0, title=f"Drone {name} Positions")
        plot_velocity(track_t, track_state[:, 3:6], track_ref[:, 3:6], takeoff_line=None, id=0, title=f"Drone {name} Velocity")
        # plot_angles(t, angles, controls, takeoff_line=takeoff_line, eps_max=0.5, title=f"Drone {name} Angles")
        # plot_angles(t[0:len(pose_angles)], pose_angles, controls[0:len(pose_angles), :], eps_max=0.5, title=f"Drone {name} Angles")

        plot_controls(track_t[0:len(track_controls)], track_angles, track_controls, takeoff_line=None, Tmax=17, eps_max=0.2, id=0, title=f"Drone {name} Controls")

        plot_v(track_t, track_v, track_vref, id=0, title=f"Drone {name} Accelerations")
    
    
    # load_data = np.load('./drone_data_{}_seq.npy'.format(sim_id),allow_pickle=True).item()
    # data = load_data['result']
    # ref = load_data['parameter']
    # # print(ref)
    # nb = len(data)
    # print("number of bodies = ",nb)
    # xref = {}
    # yref = {}
    # zref = {}
    # vxref = {}
    # vyref = {}
    # vzref = {}
    
    # # for i in uris:
    # #     xref[i] = ref[i][:,0]
    # #     yref[i] = ref[i][:,1]
    # #     zref[i] = ref[i][:,2]
    # #     vxref[i] = ref[i][:,3]
    # #     vyref[i] = ref[i][:,4]
    # #     vzref[i] = ref[i][:,5]
    # # for i in uris:
    # #     xref[i] = ref['trajectory'][0,:]
    # #     yref[i] = ref['trajectory'][1,:]
    # #     zref[i] = ref['trajectory'][2,:]
    # #     vxref[i] = ref['trajectory'][3,:]
    # #     vyref[i] = ref['trajectory'][4,:]
    # #     vzref[i] = ref['trajectory'][5,:]
    # for i in uris:
    #     xref[i] = ref[i][0,:]
    #     yref[i] = ref[i][1,:]
    #     zref[i] = ref[i][2,:]
    #     vxref[i] = ref[i][3,:]
    #     vyref[i] = ref[i][4,:]
    #     vzref[i] = ref[i][5,:]
    # x = {}
    # y = {}
    # z = {}
    # vx = {}
    # vy = {}
    # vz = {}
    # T = {}
    # Roll = {}
    # Pitch = {}
    # Yaw = {}
    # time = {}
    # for i in list_of_bodies:
    #     x[i] = data[i][0::11]
    #     y[i] = data[i][1::11]
    #     z[i] = data[i][2::11]
    #     vx[i] = data[i][3::11]
    #     vy[i] = data[i][4::11]
    #     vz[i] = data[i][5::11]
    #     T[i] = data[i][6::11]
    #     Roll[i] = data[i][7::11]
    #     Pitch[i] = data[i][8::11]
    #     Yaw[i] = data[i][9::11]
    #     time[i] = data[i][10::11]

    # print(time)
    # acc_x = {}
    # acc_y = {}
    # acc_z = {}
    # for i in list_of_bodies:
    #     acc_x[i] = T[i] * (np.cos(Roll[i])*np.sin(Pitch[i])*np.cos(Yaw[i]/180*np.pi) +
    #                         np.sin(Roll[i])*np.sin(Yaw[i]/180*np.pi)  )
    #     acc_y[i] = T[i] * (np.cos(Roll[i])*np.sin(Pitch[i])*np.sin(Yaw[i]/180*np.pi) -
    #                         np.sin(Roll[i])*np.cos(Yaw[i]/180*np.pi)  )
    #     acc_z[i] = T[i] * np.cos(Roll[i])*np.sin(Pitch[i]) - 9.81
    # # print(T)
    # for i in range(len(list_of_bodies)):
    #     print(T[list_of_bodies[i]][len(T[list_of_bodies[i]])//2])
    #     print(z[list_of_bodies[i]][len(z[list_of_bodies[i]])//2]-zref[uris[i]][len(zref[uris[i]])//2])
    #     print(vz[list_of_bodies[i]][len(vz[list_of_bodies[i]])//2]-vzref[uris[i]][len(vzref[uris[i]])//2])
    #     print(x[list_of_bodies[i]][len(x[list_of_bodies[i]])//2]-xref[uris[i]][len(xref[uris[i]])//2])
    #     print(vx[list_of_bodies[i]][len(vx[list_of_bodies[i]])//2]-vxref[uris[i]][len(vxref[uris[i]])//2])
    #     print(y[list_of_bodies[i]][len(y[list_of_bodies[i]])//2]-yref[uris[i]][len(yref[uris[i]])//2])
    #     print(vy[list_of_bodies[i]][len(vy[list_of_bodies[i]])//2]-vyref[uris[i]][len(vyref[uris[i]])//2])

    # print('Timestamps : ',len(time[list_of_bodies[0]]))
    # # print(time[list_of_bodies[0]][0])
    # # print(time[list_of_bodies[0]][149])
    # # print(time[list_of_bodies[0]][0])
    # # print(time[list_of_bodies[0]][149])
    # # Plot position in 3D plot:
    # fig1 = plt.figure('3D tracking')
    # # Formation design
    # g0 = 0.5
    # # gap = np.zeros((3, nbr_agents))
    # gap = np.zeros((3, 4))
    # gap[0,:] = [-g0, 0, -g0, 0]
    # gap[1,:] = [g0, g0, 0, 0]
    # gap[2,:] = [0, 0, 0, 0]
    # ax1 = fig1.add_subplot(projection='3d')


    # for i in range(len(list_of_bodies)):
    #     ax1.plot(x[list_of_bodies[i]],y[list_of_bodies[i]],z[list_of_bodies[i]], '--',label='Trajectory')
    #     # ax1.scatter(xref[uris[i]][1:]+ gap[0,i],yref[uris[i]][1:]+ gap[1,i],zref[uris[i]][1:]+ gap[2,i],label='Reference')
    #     ax1.scatter(xref[uris[i]][1:],yref[uris[i]][1:],zref[uris[i]][1:],label='Reference')

        
    
    # ax1.set_xlabel('x (m)')
    # ax1.set_ylabel('y (m)')
    # ax1.set_zlabel('z (m)')
    # ax1.legend()
    
    # fig12 = plt.figure('z(t)')
    # for i in range(len(list_of_bodies)):  
    #     ax1 = fig12.add_subplot(nb,1,i+1)
    #     ax1.plot(time[list_of_bodies[0]],z[list_of_bodies[i]],label='Trajectory')
    #     ax1.plot(time[list_of_bodies[0]],zref[uris[i]][1:],label='Reference')
        
    #     ax1.grid(True)
    #     ax1.set_xlabel("Time (s)")
    #     ax1.set_ylabel(f"z {list_of_bodies[i]} (m)")
    #     if (i==0):
    #         ax1.legend()
    
    
    # fig13 = plt.figure('x(t)')
    # for i in range(len(list_of_bodies)):  
    #     ax1 = fig13.add_subplot(nb,1,i+1)
    #     # ax1.plot(time[list_of_bodies[i]],x[list_of_bodies[i]])
    #     # ax1.plot(time[list_of_bodies[i]],xref[uris[i]])
    #     ax1.plot(time[list_of_bodies[0]],x[list_of_bodies[i]],label='Trajectory')
    #     ax1.plot(time[list_of_bodies[0]],xref[uris[i]][1:],label='Reference')
    #     ax1.grid(True)
    #     ax1.set_xlabel("Time (s)")
    #     ax1.set_ylabel(f"x {list_of_bodies[i]} (m)")
    #     if (i==0):
    #         ax1.legend()

    # fig14 = plt.figure('y(t)')
    # for i in range(len(list_of_bodies)):  
    #     ax1 = fig14.add_subplot(nb,1,i+1)
    #     # ax1.plot(time[list_of_bodies[i]],y[list_of_bodies[i]],label='Trajectory')
    #     # ax1.plot(time[list_of_bodies[i]],yref[uris[i]])
    #     ax1.plot(time[list_of_bodies[0]],y[list_of_bodies[i]],label='Trajectory')
    #     ax1.plot(time[list_of_bodies[0]],yref[uris[i]][1:],label='Reference')
    #     ax1.grid(True)
    #     ax1.set_xlabel("Time (s)")
    #     ax1.set_ylabel(f"y {list_of_bodies[i]} (m)")
    #     if (i==0):
    #         ax1.legend()
            
    # fig21 = plt.figure("vx(t)")
    # for i in range(len(list_of_bodies)):
    #     ax2 = fig21.add_subplot(nb,1,i+1)
    #     # ax2.plot(time[list_of_bodies[i]],vx[list_of_bodies[i]],label='Trajectory')
    #     # ax2.plot(time[list_of_bodies[i]],vxref[uris[i]])
    #     ax2.plot(time[list_of_bodies[0]],vx[list_of_bodies[i]],label='Trajectory')
    #     ax2.plot(time[list_of_bodies[0]],vxref[uris[i]][1:],label='Reference')
    #     ax2.grid(True)
    #     ax2.set_xlabel("Time (s)")
    #     ax2.set_ylabel(f"vx {list_of_bodies[i]} (m/s)")
    #     if (i==0):
    #         ax2.legend()


    # fig22 = plt.figure("vy(t)")
    # for i in range(len(list_of_bodies)):  
    #     ax2 = fig22.add_subplot(nb,1,i+1)
    #     ax2.plot(time[list_of_bodies[0]],vy[list_of_bodies[i]],label='Trajectory')
    #     ax2.plot(time[list_of_bodies[0]],vyref[uris[i]][1:],label='Reference')
    #     ax2.grid(True)
    #     ax2.set_xlabel("Time (s)")
    #     ax2.set_ylabel(f"vy {list_of_bodies[i]} (m/s)")
    #     if (i==0):
    #         ax2.legend()


    # fig23 = plt.figure("vz(t)")
    # for i in range(len(list_of_bodies)):  
    #     ax2 = fig23.add_subplot(nb,1,i+1)
    #     ax2.plot(time[list_of_bodies[0]],vz[list_of_bodies[i]],label='Trajectory')
    #     ax2.plot(time[list_of_bodies[0]],vzref[uris[i]][1:],label='Reference')
    #     ax2.grid(True)
    #     ax2.set_xlabel("Time (s)")
    #     ax2.set_ylabel(f"vz {list_of_bodies[i]} (m/s)")
    #     if (i==0):
    #         ax2.legend()   

    # # accel
    # fig21b = plt.figure("ax(t)")
    # for i in range(len(list_of_bodies)):
    #     ax2b = fig21b.add_subplot(nb,1,i+1)
    #     # ax2.plot(time[list_of_bodies[i]],vx[list_of_bodies[i]],label='Trajectory')
    #     # ax2.plot(time[list_of_bodies[i]],vxref[uris[i]])
    #     ax2b.plot(time[list_of_bodies[0]],acc_x[list_of_bodies[i]],label='Trajectory')
    #     # ax2b.plot(time[list_of_bodies[0]],vxref[uris[i]][1:]+gap_acc[0,i,:],label='Reference')
    #     ax2b.grid(True)
    #     ax2b.set_xlabel("Time (s)")
    #     ax2b.set_ylabel(f"ax {list_of_bodies[i]} (m/s2)")
    #     if (i==0):
    #         ax2b.legend()


    # fig22b = plt.figure("ay(t)")
    # for i in range(len(list_of_bodies)):  
    #     ax2b = fig22b.add_subplot(nb,1,i+1)
    #     ax2b.plot(time[list_of_bodies[0]],acc_y[list_of_bodies[i]],label='Trajectory')
    #     # ax2b.plot(time[list_of_bodies[0]],vyref[uris[i]][1:]+gap_acc[0,i,:],label='Reference')
    #     ax2b.grid(True)
    #     ax2b.set_xlabel("Time (s)")
    #     ax2b.set_ylabel(f"ay {list_of_bodies[i]} (m/s2)")
    #     if (i==0):
    #         ax2b.legend()


    # fig23b = plt.figure("az(t)")
    # for i in range(len(list_of_bodies)):  
    #     ax2b = fig23b.add_subplot(nb,1,i+1)
    #     ax2b.plot(time[list_of_bodies[0]],acc_z[list_of_bodies[i]],label='Trajectory')
    #     # ax2b.plot(time[list_of_bodies[0]],vzref[uris[i]][1:],label='Reference')
    #     ax2b.grid(True)
    #     ax2b.set_xlabel("Time (s)")
    #     ax2b.set_ylabel(f"az {list_of_bodies[i]} (m/s2)")
    #     if (i==0):
    #         ax2b.legend() 

    # fig3 = plt.figure("Thrust T(t)")
    # for i in range(len(list_of_bodies)):
    #     ax3 = fig3.add_subplot(nb,1,i+1)
    #     ax3.plot(time[list_of_bodies[0]],T[list_of_bodies[i]])
    #     ax3.grid(True)
    #     ax3.set_xlabel("Time (s)")
    #     ax3.set_ylabel(f"Thrust {list_of_bodies[i]} (m/s^{2})")

    # fig32 = plt.figure("Roll \phi(t)")
    # for i in range(len(list_of_bodies)):
    #     ax3 = fig32.add_subplot(nb,1,i+1)
    #     ax3.plot(time[list_of_bodies[0]],Roll[list_of_bodies[i]]*180/np.pi)
    #     ax3.grid(True)
    #     ax3.set_xlabel("Time (s)")
    #     ax3.set_ylabel(f"Roll {list_of_bodies[i]} (°)")

    # fig33 = plt.figure("Pitch theta(t)")
    # for i in range(len(list_of_bodies)):
    #     ax3 = fig33.add_subplot(nb,1,i+1)
    #     ax3.plot(time[list_of_bodies[0]],Pitch[list_of_bodies[i]]*180/np.pi)
    #     ax3.grid(True)
    #     ax3.set_xlabel("Time (s)")
    #     ax3.set_ylabel(f"Pitch {list_of_bodies[i]} (°)")
    
    # fig34 = plt.figure("Yaw psi(t)")
    # for i in range(len(list_of_bodies)):
    #     ax3 = fig34.add_subplot(nb,1,i+1)
    #     ax3.plot(time[list_of_bodies[0]],Yaw[list_of_bodies[i]])
    #     ax3.grid(True)
    #     ax3.set_xlabel("Time (s)")
    #     ax3.set_ylabel(f"Yaw {list_of_bodies[i]} (°)")

    # dc=0.1
    # a1=6
    # a2=8
    # simulator={}
    # simulator['h_up'] = np.zeros((3,len(xref[uris[0]])-1,len(list_of_bodies)))
    # simulator['h_down'] = np.zeros((3,len(xref[uris[0]])-1,len(list_of_bodies)))
    # if len(list_of_bodies)>1:
    #     for i in range(len(list_of_bodies)):
    #         simulator["h_up"][0,:,i] = dc + xref[uris[i]][1:] - x[list_of_bodies[i]] + gap[0,i]
    #         simulator["h_down"][0,:,i] = dc - xref[uris[i]][1:] + x[list_of_bodies[i]] - gap[0,i]
    #         simulator["h_up"][1,:,i] = dc + yref[uris[i]][1:] - y[list_of_bodies[i]] + gap[1,i]
    #         simulator["h_down"][1,:,i] = dc - yref[uris[i]][1:] + y[list_of_bodies[i]] - gap[1,i]
    #         simulator["h_up"][2,:,i] = dc + zref[uris[i]][1:] - z[list_of_bodies[i]] + gap[2,i]
    #         simulator["h_down"][2,:,i] = dc - zref[uris[i]][1:] + z[list_of_bodies[i]] - gap[2,i]
    # else:
    #     for i in range(len(list_of_bodies)):
    #         simulator["h_up"][0,:,i] = dc + xref[uris[i]][1:] - x[list_of_bodies[i]] + gap[0,-1]
    #         simulator["h_down"][0,:,i] = dc - xref[uris[i]][1:] + x[list_of_bodies[i]] - gap[0,-1]
    #         simulator["h_up"][1,:,i] = dc + yref[uris[i]][1:] - y[list_of_bodies[i]] + gap[1,-1]
    #         simulator["h_down"][1,:,i] = dc - yref[uris[i]][1:] + y[list_of_bodies[i]] - gap[1,-1]
    #         simulator["h_up"][2,:,i] = dc + zref[uris[i]][1:] - z[list_of_bodies[i]] + gap[2,-1]
    #         simulator["h_down"][2,:,i] = dc - zref[uris[i]][1:] + z[list_of_bodies[i]] - gap[2,-1]        

    # fighqx = plt.figure()
    # for i in range(len(list_of_bodies)):
    #     axhqx = fighqx.add_subplot(len(list_of_bodies),1,i+1)
    #     axhqx.plot(time[list_of_bodies[0]],simulator['h_up'][0,:,i],label='CBF-QP + Nominal')
    #     # axhqx.plot(time[list_of_bodies[0]],simulator['hn_up'][0,:,i],label='Nominal')
    #     axhqx.grid(visible=True)

    #     axhqx.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * dc ,'k--',linewidth=1.5)
    #     axhqx.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * -dc ,'k--',linewidth=1.5)
    #     axhqx.set_xlabel('Time (s)')
    #     axhqx.set_ylabel('hx (m)')
    #     if (i==0):
    #         axhqx.legend()

    # fighqy = plt.figure()
    # for i in range(len(list_of_bodies)):
    #     axhqy = fighqy.add_subplot(len(list_of_bodies),1,i+1)
    #     axhqy.plot(time[list_of_bodies[0]],simulator['h_up'][1,:,i],label='CBF-QP + Nominal')
    #     axhqy.grid(visible=True)

    #     axhqy.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * dc * 2,'k--',linewidth=1.5)
    #     axhqy.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * -dc * 2,'k--',linewidth=1.5)
    #     axhqy.set_xlabel('Time (s)')
    #     axhqy.set_ylabel('hy (m)')
    #     if (i==0):
    #         axhqy.legend()

    # fighqz = plt.figure()
    # for i in range(len(list_of_bodies)):
    #     axhqz = fighqz.add_subplot(len(list_of_bodies),1,i+1)
    #     axhqz.plot(time[list_of_bodies[0]],simulator['h_up'][2,:,i],label='CBF-QP + Nominal')

    #     axhqz.grid(visible=True)

    #     axhqz.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * dc * 2,'k--',linewidth=1.5)
    #     axhqz.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * -dc * 2,'k--',linewidth=1.5)
    #     axhqz.set_xlabel('Time (s)')
    #     axhqz.set_ylabel('hz (m)')
    #     if (i==0):
    #         axhqz.legend()

    # fighqx_ = plt.figure()
    # for i in range(len(list_of_bodies)):
    #     axhqx_ = fighqx_.add_subplot(len(list_of_bodies),1,i+1)
    #     axhqx_.plot(time[list_of_bodies[0]],simulator['h_down'][0,:,i],label='CBF-QP + Nominal')

    #     axhqx_.grid(visible=True)

    #     axhqx_.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * dc ,'k--',linewidth=1.5)
    #     axhqx_.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * -dc ,'k--',linewidth=1.5)
    #     axhqx_.set_xlabel('Time (s)')
    #     axhqx_.set_ylabel('hx_ (m)')
    #     if (i==0):
    #         axhqx_.legend()

    # fighqy_ = plt.figure()
    # for i in range(len(list_of_bodies)):
    #     axhqy_ = fighqy_.add_subplot(len(list_of_bodies),1,i+1)
    #     axhqy_.plot(time[list_of_bodies[0]],simulator['h_down'][1,:,i],label='CBF-QP + Nominal')

    #     axhqy_.grid(visible=True)

    #     axhqy_.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * dc * 2,'k--',linewidth=1.5)
    #     axhqy_.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * -dc * 2,'k--',linewidth=1.5)
    #     axhqy_.set_xlabel('Time (s)')
    #     axhqy_.set_ylabel('hy_ (m)')
    #     if (i==0):
    #         axhqy_.legend()

    # fighqz_ = plt.figure()
    # for i in range(len(list_of_bodies)):
    #     axhqz_ = fighqz_.add_subplot(len(list_of_bodies),1,i+1)
    #     axhqz_.plot(time[list_of_bodies[0]],simulator['h_down'][2,:,i],label='CBF-QP + Nominal')

    #     axhqz_.grid(visible=True)

    #     axhqz_.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * dc * 2,'k--',linewidth=1.5)
    #     axhqz_.plot(time[list_of_bodies[0]],np.ones(np.size(time[list_of_bodies[0]])) * -dc * 2,'k--',linewidth=1.5)
    #     axhqz_.set_xlabel('Time (s)')
    #     axhqz_.set_ylabel('hz_ (m)')
    #     if (i==0):
    #         axhqz_.legend()

    ## plot inter-agent distances ##
    fig_inter = plt.figure("Inter-agent distances plot",)
    ax = fig_inter.add_subplot(111)
    ax.grid(True)
    inter_plot = {}
    for i in range(nb-1):
        inter_plot[i] = {}
        for j in range(i+1,nb):
            # inter_plot[i][j], = ax.plot(ref['time_step'][1:len(ref['time_step'])],simulator['dist_ij'][i,j,:], label=f"(i,j)=({i},{j})")
            inter_plot[i][j], = ax.plot(track_t,simu['dist_ij'][i,j,:], label=f"(i,j)=({i},{j})")
            
            plt.setp(inter_plot[i][j], linestyle='-', linewidth=2.0)
    
    ax.set_xlabel(r"Time $t(s)$", usetex=False, fontsize=12)
    ax.set_ylabel(r"$d_{ij}(t)$", usetex=False, fontsize=12)
    ax.set_title(r"Inter-agent distance", usetex=False, fontsize=14)


    rd_params = {}
    rd_params['ts'] = 0.1           # sampling time of replicator dynamics
    rd_params['nbr_steps'] = 10     # total rd steps 
    rd_params['rho'] = 0.7         # maximum distance of connectivity  
    view_population = 1
    view_pos = 1

    # Nb = len(time[list_of_bodies[0]])
    Nb = len(track_t)
    # # t_rd = np.linspace(0,Tf,Nb*rd_params['nbr_steps'])
    # t_rd = np.linspace(0,time[list_of_bodies[0]][-1],Nb*rd_params['nbr_steps'])

    t_rd = np.linspace(0,track_t[-1],Nb*rd_params['nbr_steps'])
    if view_population:
        fig_p = plt.figure("Population distribution plot",)
        dimension_traj = 3
        p_labels = ['p_{x}','p_{y}','p_{z}']
        p_ref_plot = {}
        p_plot = {}

        for j in range(dimension_traj):
            ax = fig_p.add_subplot(dimension_traj,1,j+1)
            ax.grid(True)

            # acc_ref_plot[j], = ax.plot(ref['time_step'], ref['v_ref'][j,:])

            # plt.setp(acc_ref_plot[j], color='m', linestyle='--', linewidth=3.5)

            p_plot[j] = {}
            for i in range(1,nb):
                p_plot[j][i], = ax.plot(t_rd, simu['p_sim'][j,i,1:(rd_params['nbr_steps']+1),:].flatten(order='F'))
                # plt.setp(p_plot[j][i], color=colors[i], linestyle='-', linewidth=3.0)

            if j==0:
                ax.set_title(r"Population $p=p(t)$", usetex=False, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=False, fontsize=12)
            ax.set_ylabel(f"${p_labels[j]}(t)$", usetex=False, fontsize=12)

    # if view_population:
    #     fig_dp = plt.figure("Population dynamics plot",)
    #     dimension_traj = 3
    #     dp_labels = ['\\frac{dp_{x}}{dt}','\\frac{dp_{y}}{dt}','\\frac{dp_{y}}{dt}']
    #     dp_ref_plot = {}
    #     dp_plot = {}

    #     for j in range(dimension_traj):
    #         ax = fig_dp.add_subplot(dimension_traj,1,j+1)
    #         ax.grid(True)

    #         # acc_ref_plot[j], = ax.plot(ref['time_step'], ref['v_ref'][j,:])

    #         # plt.setp(acc_ref_plot[j], color='m', linestyle='--', linewidth=3.5)

    #         dp_plot[j] = {}
    #         for i in range(nb):
    #             dp_plot[j][i], = ax.plot(t_rd, simu['dp_sim'][j,i,:,:].flatten(order='F'))
    #             # plt.setp(dp_plot[j][i], color=colors[i], linestyle='-', linewidth=3.0)

    #         if j==0:
    #             ax.set_title(r"$\dot{\mathbf{p}}=diag(\mathbf{p})( diag(\mathbf{F})\mathbf{A}  \mathbf{p} - \mathbf{A} diag(\mathbf{F}) \mathbf{p} )$", usetex=True, fontsize=14)
    #         ax.set_xlabel(r"Time $t(s)$", usetex=True, fontsize=12)
    #         ax.set_ylabel(fr"${dp_labels[j]}(t)$", usetex=True, fontsize=12)

    # if view_population:
    #     fig_F = plt.figure("Fitness functions plot",)
    #     dimension_traj = 3
    #     F_labels = ['F_{x}','F_{y}','F_{z}']
    #     F_ref_plot = {}
    #     F_plot = {}

    #     for j in range(dimension_traj):
    #         ax = fig_F.add_subplot(dimension_traj,1,j+1)
    #         ax.grid(True)

    #         # acc_ref_plot[j], = ax.plot(ref['time_step'], ref['v_ref'][j,:])

    #         # plt.setp(acc_ref_plot[j], color='m', linestyle='--', linewidth=3.5)

    #         F_plot[j] = {}
    #         for i in range(nb):
    #             F_plot[j][i], = ax.plot(t_rd, simu['F_sim'][i,j,:,:].flatten(order='F'))
    #             # plt.setp(F_plot[j][i], color=colors[i], linestyle='-', linewidth=3.0)

    #         if j==0:
    #             ax.set_title(r"Fitness functions on axes", usetex=True, fontsize=14)
    #         ax.set_xlabel(r"Time $t(s)$", usetex=True, fontsize=12)
    #         ax.set_ylabel(f"${F_labels[j]}(t)$", usetex=True, fontsize=12)
    
    # Replicator dynamics and reference plot
    if view_pos:
        fig_ref = plt.figure("Replicator dynamics and reference plot",)
        dimension_traj = 3
        pos_labels = ['x','y','z']
        rep_ref_plot = {}
        rep_plot = {}

        for m in range(dimension_traj):
            ax = fig_ref.add_subplot(dimension_traj,1,m+1)
            ax.grid(True)

            # rep_ref_plot[m], = ax.plot(track_t, ref[uris[-1]][m,1:])

            # plt.setp(rep_ref_plot[m], color='m', linestyle='--', linewidth=3.5)

            rep_plot[m] = {}
            for i in range(nb):
                rep_plot[m][i], = ax.plot(track_t, simu['xref_sim'][m,i,:])
                # plt.setp(rep_plot[m][i], color=colors[i], linestyle='-', linewidth=3.0)
                
            if m==0:
                ax.set_title(r"Rep Positions on $x$, $y$ and $z$-axis", usetex=False, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=False, fontsize=12)
            ax.set_ylabel(f"${pos_labels[m]}(t)$", usetex=False, fontsize=12)

    # # Difference between leader's reference + gaps vs result of replicator dynamics
    # view_diff = 1
    # if view_diff:
    #     fig_diff = plt.figure("Difference between replicator dynamics and reference",)
    #     dimension_traj = 3
    #     pos_labels = ['\delta x','\delta y','\delta z']
    #     diff_plot = {}
    #     # rep_plot = {}

    #     for m in range(dimension_traj):
    #         if (m==0):
    #             ax.legend()
    #         ax = fig_diff.add_subplot(dimension_traj,1,m+1)
    #         ax.grid(True)

    #         # rep_ref_plot[m], = ax.plot(time[list_of_bodies[0]], ref[uris[-1]][1:,m] - simu["xref_sim"][m,i,:])

    #         # plt.setp(rep_ref_plot[m], color='m', linestyle='--', linewidth=3.5)

    #         diff_plot[m] = {}
    #         for i in range(nb):
    #             diff_plot[m][i], = ax.plot(time[list_of_bodies[0]], ref[uris[-1]][m,1:] + gap[m,i] - simu["xref_sim"][m,i,:])
    #             # plt.setp(diff_plot[m][i], color=colors[i], linestyle='-', linewidth=3.0)
                
    #         if m==0:
    #             ax.set_title(r"Diff Rep vs Ref on $x$, $y$ and $z$-axis", usetex=True, fontsize=14)
    #         ax.set_xlabel(r"Time $t(s)$", usetex=True, fontsize=12)
    #         ax.set_ylabel(f"${pos_labels[m]}(t)$", usetex=True, fontsize=12)


    plt.show()