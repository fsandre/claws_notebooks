import numpy as np
import control as ct

def include_tasdot_state(ss_lon, i_tas_state, state_label='tasdot'):
    A_stab_2 = np.dot(ss_lon.A, ss_lon.A)
    AB_stab = np.dot(ss_lon.A, ss_lon.B)
    A_tasdot_col = np.c_[ss_lon.A, np.zeros(ss_lon.A.shape[0])]
    A_tasdot = np.r_[A_tasdot_col, [[*A_stab_2[i_tas_state], 0]]]
    B_tasdot = np.r_[ss_lon.B, [AB_stab[i_tas_state]]]
    C_tasdot_col = np.c_[ss_lon.C, np.zeros(ss_lon.C.shape[0])]
    C_tasdot = np.r_[C_tasdot_col, [[*np.zeros(ss_lon.C.shape[1]), 1]]]
    D_tasdot = np.r_[ss_lon.D, np.zeros([1, ss_lon.D.shape[1]])]
    input_names = ss_lon.input_labels
    output_names = ss_lon.output_labels
    state_names = ss_lon.state_labels
    state_names.append(state_label)
    output_names.append(state_label)
    return ct.ss(A_tasdot, B_tasdot, C_tasdot, D_tasdot, 
                inputs=input_names, states=state_names, outputs=output_names)

def include_alpha_int_state(ss_lon, i_alpha_state, state_label='int_alpha'):
    A_int_col = np.c_[ss_lon.A, np.zeros(ss_lon.A.shape[0])]
    A_alpha_row = np.zeros(ss_lon.A.shape[1] + 1)
    A_alpha_row[i_alpha_state] = 1
    A_int = np.r_[A_int_col, [A_alpha_row]]
    B_int = np.r_[ss_lon.B, np.zeros([1, ss_lon.B.shape[1]])]
    C_int_col = np.c_[ss_lon.C, np.zeros(ss_lon.C.shape[0])]
    C_int = np.r_[C_int_col, [[*np.zeros(ss_lon.C.shape[1]), 1]]]
    D_int = np.r_[ss_lon.D, np.zeros([1, ss_lon.D.shape[1]])]
    input_names = ss_lon.input_labels
    output_names = ss_lon.output_labels
    state_names = ss_lon.state_labels
    state_names.append(state_label)
    output_names.append(state_label)
    return ct.ss(A_int, B_int, C_int, D_int, 
                inputs=input_names, states=state_names, outputs=output_names)

def include_alpha_cmd(ss_cl, i_int_alpha_state):
    B = np.c_[ss_cl.B, np.zeros([ss_cl.B.shape[0],1])]
    D = np.c_[ss_cl.D, np.zeros([ss_cl.D.shape[0],1])]
    B[i_int_alpha_state][-1] = -1
    return ct.ss(ss_cl.A, B, ss_cl.C, D, 
                 inputs=ss_cl.input_labels + ['alpha_cmd_rad'],
                 states=ss_cl.state_labels,
                 outputs=ss_cl.output_labels)

def aoa_control_cl(ss_ac, params):
    i_v_state = ss_ac.state_index['V_fps']
    i_alpha_state = ss_ac.state_index['alpha_rad']
    ss_vdot = include_tasdot_state(ss_ac, i_v_state, 'tasdot_fps2')
    ss_vdot_int_alpha = include_alpha_int_state(ss_vdot, i_alpha_state, 'int_alpha_rad')
    ss_ol_elev = ss_vdot_int_alpha
    i_int_alpha_state = ss_ol_elev.state_index['int_alpha_rad']

    ss_Kc = ct.ss([], [], [], np.array([
                                         [0]*9,
                                         [-params['KV'], -params['KP'], params['Ktheta'], params['Kq'], 0, 0, 0, -params['Ktasdot'], params['Kint']]
                                       ])/np.pi*180,
                inputs=['V_fb_fps', 'alpha_fb_rad', 'theta_fb_rad', 'q_fb_rps', 'nx_fb_g', 'ny_fb_g', 'nz_fb_g', 'tasdot_fb_fps2', 'int_alpha_fb_rad'],
                outputs=['throttle_out_u', 'elev_out_deg'])
    ss_Kc.D.shape = (2,9)
    ss_cl_fb = ss_ol_elev.feedback(ss_Kc, 1)
    ss_cl = include_alpha_cmd(ss_cl_fb, i_int_alpha_state)
    ss_cl.D.shape = (9,3)
    ss_cl.set_outputs(ss_ol_elev.output_labels)
    return ss_cl
