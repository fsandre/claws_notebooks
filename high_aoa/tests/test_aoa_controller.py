import unittest
import aoa_controller
import control as ct
import numpy as np
import matplotlib.pyplot as plt

class TestAOAController(unittest.TestCase):

    def test_include_vdot(self):
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
        ss_vdot = aoa_controller.include_tasdot_state(ss_civil_ac, i_v_state, 'tasdot_fps2')
        T, yout = ct.step_response(ss_vdot, T=5.0)
        i_v_output = ss_vdot.output_index['V_fps']
        i_vdot_output = ss_vdot.output_index['tasdot_fps2']
        i_elev_input = ss_vdot.input_index['elev_deg']
        y_vdot_out = yout[i_vdot_output][i_elev_input][:-1]
        y_vdot_exp = np.diff(yout[i_v_output][i_elev_input])/np.diff(T)
        err_vdot = np.max(np.abs(y_vdot_out-y_vdot_exp))
        print(err_vdot)
        self.assertTrue(err_vdot<1e-2)
        plt.plot(T, yout[i_vdot_output][i_elev_input], T[0:-1], np.diff(yout[i_v_output][i_elev_input])/np.diff(T), 'r--')
        plt.legend(['Vdot state [fps2]', 'diff(V)/diff(T)'])
        plt.grid(True)
        plt.show()

    def test_include_alpha_int(self):
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

        i_alpha_state = ss_civil_ac.state_index['alpha_rad']
        ss_int_alpha = aoa_controller.include_alpha_int_state(ss_civil_ac, i_alpha_state, 'int_alpha_rad')
        T, yout = ct.step_response(ss_int_alpha, T=5.0)
        i_alpha_output = ss_int_alpha.output_index['alpha_rad']
        i_alpha_int_output = ss_int_alpha.output_index['int_alpha_rad']
        i_elev_input = ss_int_alpha.input_index['elev_deg']
        y_ialpha_out = np.diff(yout[i_alpha_int_output][i_elev_input])/np.diff(T)
        y_ialpha_exp = yout[i_alpha_output][i_elev_input][:-1]
        err_ialpha = np.max(np.abs(y_ialpha_out-y_ialpha_exp))
        print(err_ialpha)
        self.assertTrue(err_ialpha<1e-3)
        plt.plot(T[0:-1], np.diff(yout[i_alpha_int_output][i_elev_input])/np.diff(T), T, yout[i_alpha_output][i_elev_input], 'r--')
        plt.legend(['diff(alpha)/diff(T)', 'alpha[rad]'])
        plt.grid(True)
        plt.show()

if __name__=='__main__':
    unittest.main()
