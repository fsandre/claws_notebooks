import numpy as np
import control as ct
import aoa_controller
import matplotlib.pyplot as plt

def aoa_fb_analysis(ss_ol_elev=None, **kwargs):
    max_wn = 0
    ss_cl = aoa_controller.aoa_control_cl(ss_ol_elev, kwargs)
    ss_cl_aoa_cmd = ss_cl[list(range(ss_cl.noutputs)), ss_cl.input_index['alpha_cmd_rad']]
    ss_cl_aoa_cmd.D.shape = (9,1)
    ss_cl_aoa_cmd = np.pi/180*ss_cl_aoa_cmd #converting input to degrees
    T, yout = ct.step_response(ss_cl_aoa_cmd, T=100.0)

    wn_zeta_ol = ss_ol_elev.damp()
    wn_zeta = ss_cl.damp()
    max_wn = max(abs(np.real(np.r_[wn_zeta[2], wn_zeta_ol[2]])))
    fig, axs = plt.subplots(2,2)
    fig.set_figwidth(10)
    fig.set_figheight(4)
    fig.tight_layout()
    
    axs[0,0].plot(np.real(wn_zeta[2]), np.imag(wn_zeta[2]), 'ro', np.real(wn_zeta_ol[2]), np.imag(wn_zeta_ol[2]), 'bx')
    axs[0,0].grid(True)
    y_wn_0_7 = max_wn*np.tan(np.arccos(0.7))
    axs[0,0].plot([0, -max_wn], [0,  y_wn_0_7],'b--')
    axs[0,0].plot([0, -max_wn], [0, -y_wn_0_7],'b--')
    axs[0,0].text(-0.9*max_wn, 1.1*y_wn_0_7, r'$\zeta = 0.7$', fontsize=10, color='grey')
    axs[0,0].text(-0.9*max_wn, -1.1*y_wn_0_7, r'$\zeta = 0.7$', fontsize=10, color='grey')
    
    axs[0,1].plot(T, 180/np.pi*yout[ss_cl.output_index['alpha_rad']][0])
    axs[0,1].grid(True)
    axs[0,1].set_title('Alpha[deg]')

    axs[1,0].plot(T, yout[ss_cl.output_index['V_fps']][0])
    axs[1,0].grid(True)
    axs[1,0].set_title('V[ft/s]')

    axs[1,1].plot(T, 180/np.pi*yout[ss_cl.output_index['theta_rad']][0])
    axs[1,1].grid(True)
    axs[1,1].set_title('Theta[deg]')

def aoa_sensitivity_kv(ss_ol=None, kv_range=[0.0, 0.04], **kwargs):
    gain_range = np.arange(start=kv_range[0], stop=kv_range[1], step=(kv_range[1]-kv_range[0])/5.0)
    fig, axs = plt.subplots(2,2)
    fig.set_figwidth(10)
    fig.set_figheight(4)
    fig.tight_layout()
    
    legend_str = []
    max_wn = 0
    for i_k in range(len(gain_range)):
        gain_curr = gain_range[i_k]
        kwargs['KV'] = gain_curr
        ss_cl = aoa_controller.aoa_control_cl(ss_ol, kwargs)
        ss_cl_aoa_cmd = ss_cl[list(range(ss_cl.noutputs)), ss_cl.input_index['alpha_cmd_rad']]
        ss_cl_aoa_cmd.D.shape = (9,1) 
        ss_cl_aoa_cmd = np.pi/180*ss_cl_aoa_cmd #converting input to degrees
        T, yout = ct.step_response(ss_cl_aoa_cmd, T=100.0)
        legend_str.append('KV={:.3f}'.format(gain_curr))

        rgb_color = (0.8, 0.8-i_k/10.0, 0.0)
        wn_zeta = ss_cl.damp()
        max_wn = max([max_wn, *abs(np.real(wn_zeta[2]))])
        axs[0,0].plot(np.real(wn_zeta[2]), np.imag(wn_zeta[2]), 'x',
                      color=rgb_color)
    
        axs[0,1].plot(T, 180/np.pi*yout[ss_cl.output_index['alpha_rad']][0],
                      color=rgb_color)
        axs[0,1].grid(True)
        axs[0,1].set_title('Alpha[deg]')
    
        axs[1,0].plot(T, yout[ss_cl.output_index['V_fps']][0],
                      color=rgb_color)
        axs[1,0].grid(True)
        axs[1,0].set_title('V[ft/s]')
    
        axs[1,1].plot(T, 180/np.pi*yout[ss_cl.output_index['theta_rad']][0],
                      color=rgb_color)
        axs[1,1].grid(True)
        axs[1,1].set_title('Theta[deg]')
    
    y_wn_0_7 = max_wn*np.tan(np.arccos(0.7))
    axs[0,0].plot([0, -max_wn], [0,  y_wn_0_7],'b--')
    axs[0,0].plot([0, -max_wn], [0, -y_wn_0_7],'b--')
    axs[0,0].text(-0.9*max_wn, 1.1*y_wn_0_7, r'$\zeta = 0.7$', fontsize=10, color='grey')
    axs[0,0].text(-0.9*max_wn, -1.1*y_wn_0_7, r'$\zeta = 0.7$', fontsize=10, color='grey')
    axs[0,0].grid(True)
    axs[0,1].legend(legend_str, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)