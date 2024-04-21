import numpy as np
import control as ct
import aoa_controller
import matplotlib.pyplot as plt

A_stab = [[-0.0152827,   18.832685,  -32.170158,  -0.0001952],
          [  -0.0009885,  -0.6354411,  -0.000053,    1.0000305],
          [   1.799e-17,   0.       ,   0.      ,    1.       ],
          [   0.0000851,  -0.7754814,   0.0000045,  -0.5297807]]
B_stab = [[  9.9684819,  -0.0014153],
          [   -0.0065569,  -0.0000725],
          [   -2.562e-17,  -2.498e-17],
          [    0.0255867,  -0.0110233]]
C_stab = [[  0.9918411,   0.0146068,   0.0159216,  -0.0055678],
             [ 0.0000033,   1.0000155,   0.0000046,  -0.0000132],
             [-0.0000097,  -0.0000165,   0.9999907,   0.0000077],
              [1.799e-17,   0.,          0.       ,   1.       ],
              [0.000773,    1.373903,    0.0000622,  -0.0000532],
              [0.       ,   0.      ,    0.       ,   0.       ],
              [0.0075212,   4.7789539,  -0.0001694,  -0.0004403]]
D_stab = [[   -0.0052791,   0.0068775],
          [     -0.0000122,   0.000009], 
          [     -0.0000273,   0.0000117],
          [     -2.562e-17,  -2.498e-17],
          [      0.3140201,   0.0000631],
          [      0.       ,   0.       ],
          [      0.0004989,   0.0008925]]
input_names = ['throttle_u', 'elev_deg']
state_names = ['V_fps', 'alpha_rad', 'theta_rad', 'q_rps']
output_names = ['V_fps', 'alpha_rad', 'theta_rad', 'q_rps', 
                'nxb_g', 'nyb_g', 'nzb_g']

ss_civil_ac = ct.ss(A_stab, B_stab, C_stab, D_stab, 
                    inputs=input_names, states=state_names, outputs=output_names)

i_v_state = ss_civil_ac.state_index['V_fps']
i_alpha_state = ss_civil_ac.state_index['alpha_rad']
ss_vdot = aoa_controller.include_tasdot_state(ss_civil_ac, i_v_state, 'tasdot_fps2')
ss_vdot_int_alpha = aoa_controller.include_alpha_int_state(ss_vdot, i_alpha_state, 'int_alpha_rad')

T, yout = ct.step_response(ss_vdot_int_alpha, T=5.0)
i_v_output = ss_vdot_int_alpha.output_index['V_fps']
i_alpha_output = ss_vdot_int_alpha.output_index['alpha_rad']
i_vdot_output = ss_vdot_int_alpha.output_index['tasdot_fps2']
i_alpha_int_output = ss_vdot_int_alpha.output_index['int_alpha_rad']
i_elev_input = ss_vdot_int_alpha.input_index['elev_deg']
plt.plot(T, yout[i_vdot_output][i_elev_input], T[0:-1], np.diff(yout[i_v_output][i_elev_input])/np.diff(T), 'r--')
plt.legend(['Vdot state [fps2]', 'diff(V)/diff(T)'])
plt.grid(True)
plt.show()

plt.plot(T[0:-1], np.diff(yout[i_alpha_int_output][i_elev_input])/np.diff(T), T, yout[i_alpha_output][i_elev_input], 'r--')
plt.legend(['diff(alpha)/diff(T)', 'alpha[rad]'])
plt.grid(True)
plt.show()

ss_ol_elev = ss_vdot_int_alpha[list(range(ss_vdot_int_alpha.noutputs)),1]
ss_ol_elev.D.shape = (9,1)
i_int_alpha_state = ss_vdot_int_alpha.state_index['int_alpha_rad']

Kint = 0.551
KP = 0.1098
Kq = 0.6
KD = 0*0.6
Ktasdot = 0.009
KV = 0.007
Ktheta = 0*0.8
ss_Kc = ct.ss([], [], [], np.array([-KV, -KP, Ktheta, Kq, 0, 0, 0, -Ktasdot, Kint])/np.pi*180,
              inputs=['V_fb_fps', 'alpha_fb_rad', 'theta_fb_rad', 'q_fb_rps', 'nx_fb_g', 'ny_fb_g', 'nz_fb_g', 'tasdot_fb_fps2', 'int_alpha_fb_rad'],
              outputs=['elev_out_deg'])

legend_str = []

i_fig_resp = 1
i_fig_poles = 2
max_wn = 0
for i_k in range(10):
    gain_curr = -i_k/1000/np.pi*180
    ss_Kc_iter = ss_Kc.copy()
    ss_Kc_iter.D[0][ss_Kc_iter.input_index['V_fb_fps']] = gain_curr
    ss_cl_fb = ss_ol_elev.feedback(ss_Kc_iter,1)
    ss_cl = aoa_controller.include_alpha_cmd(ss_cl_fb, i_int_alpha_state)
    T, yout = ct.step_response(ss_cl, T=100.0)
    legend_str.append('KV={:.3f}'.format(gain_curr/180*np.pi))

    wn_zeta = ss_cl.damp()
    max_wn = max([max_wn, *abs(np.real(wn_zeta[2]))])
    plt.figure(i_fig_poles)
    f = plt.plot(np.real(wn_zeta[2]), np.imag(wn_zeta[2]), 'x')
    f[0].color=[(0.1, 0.2+i_k/100.0, 0.5)]

    plt.figure(i_fig_resp)
    f = plt.plot(T, yout[i_alpha_output][0])
    f[0].color=[(0.1, 0.2+i_k/100.0, 0.5)]

plt.figure(i_fig_resp)
plt.grid(True)
plt.title('Alpha[deg]')
plt.legend(legend_str)

plt.figure(i_fig_poles)
plt.grid(True)
plt.plot([0, -max_wn], [0,1.0202040612204073],'b--')
plt.plot([0, -max_wn], [0,-1.0202040612204073],'b--')
plt.text(-0.9*max_wn, 1.0, r'$\zeta = 0.7$', fontsize=10, color='grey')
plt.text(-0.9*max_wn, -1.0, r'$\zeta = -0.7$', fontsize=10, color='grey')
plt.show()
                    
                        
