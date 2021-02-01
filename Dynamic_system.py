import numpy as np
from casadi import *

def specifications():
    ''' Specify Problem parameters '''
    tf              = 144.      # final time
    steps           = 6        # sampling points
    x0              = np.array([0.27, 765., 0.0])
    Lsolver         = 'mumps'  #'ma97'  # Linear solver
    c_code          = False    # c_code

    # model parameter definitions
    m_param_names   = ['u_m', 'K_N', 'u_d', 'Y_nx', 'k_m', 'Kd', 'K_NL', 'Ks', 'Ki', 'Ksl', 'Kil', 'tau', 'Ka', 'L']
    m_params_vals   = [0.152, 30, 5.95e-3, 305, 0.35, 3.71e-3, 10, 142.8, 214.2, 320.6, 480.9, 0.120*1000, 0.0,  0.084]
                     

    # Define control bounds
    LI_ranges    = np.array([[100.,500.],[300.,800.],[500.,1000.],[500.,1000.]])
    FCn_ranges   = np.array([[1.,5.],[5, 10.],[1,20],[10.,20.]]) 
    cntrl_bounds = np.zeros((2,2,4,4))

    for bi in range(4):
        for bj in range(4):
            cntrl_bounds[:,:,bi,bj] = np.vstack((FCn_ranges[bj,:], LI_ranges[bi,:]))


    nx         = 3
    nu         = 2
    npsig      = 6
    npt        = len(m_param_names)
    abs_umin   = np.array([0.1, 100])
    abs_umax   = np.array([100., 1000]) 
    # define percentage parameter noise
    noise      = 0.0125         #15 5% parametric noise
    listsig    = [noise]*npsig
    for i in range(npt-npsig):
        listsig.append(0.)
    #print(listsig, len(listsig))
    cov        = np.diag(listsig)**2
    #print(cov.shape)
    #print('u_min shape', u_min.shape)
    # measurement noise: (scaled by 1e-6 in later code)
    sigma_v    = [0.4, 1e5, 0.]
    p_sig      = cov
    #print(p_sig)
    # number of MC realisations
    MCreal     = 1

    # Packing dictionary
    env_dict        = {'v_names': m_param_names, 'v_vals': m_params_vals, 'u_range': cntrl_bounds,
                        'tf': tf, 'steps': steps, 'x0': x0, 'sigma_noisev': sigma_v, 'MCreal': MCreal, 
                        'nx':nx, 'nu':nu, 'absmin': abs_umin, 'absmax': abs_umax, 'p_sig':p_sig}


    return Lsolver, c_code, env_dict
### 0.175 +-0.05

def lb_law(tau, X, Ka, z, L, I0):


    Iz = I0 * (exp(-(tau*X + Ka)*z) + exp(-(tau*X + Ka)*(L-z)))

    return Iz

def DAE_system(mparams_names, mparams_vals):
    # Define vectors with names of states
    states     = ['Cx','Cn', 'Cl']
    nd         = len(states)
    xd         = SX.sym('xd',nd)
    for i in range(nd):
        globals()[states[i]] = xd[i]

    # Define vectors with names of algebraic variables
    algebraics = []
    na         = len(algebraics)
    xa         = SX.sym('xa',na)
    for i in range(na):
        globals()[algebraics[i]] = xa[i]

    # Define vectors with names of control variables
    inputs     = ['Fnin', 'I0']
    nu         = len(inputs)
    u          = SX.sym("u",nu)
    for i in range(nu):
        globals()[inputs[i]] = u[i]

   
    # Define model parameter names and values
    modpar    = mparams_names
    modparval = mparams_vals

    nmp         = len(modpar)
    uncertainty = SX.sym('uncp', nmp)
    for i in range(nmp):
        globals()[modpar[i]] = SX(modparval[i]*(1 + uncertainty[i]))


    # Additive measurement noise
#    Sigma_v  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

    # Additive disturbance noise
#    Sigma_w  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

    # Initial additive disturbance noise
#    Sigma_w0 = [1.,150.**2,0.]*diag(np.ones(nd))*1e-3

    # Declare ODE equations (use notation as defined above)
    
    # algebraic equations
    disc_int    = 11
    Izlist      = []

    """for i in range(disc_int):
        Izlist.append(f'Iz{i}')"""

    Iz          = SX.sym("Iz",disc_int)
    for i in range(disc_int):
        #print(i, i*L/(disc_int-1), L  )
        Iz[i] = SX(lb_law(tau, Cx, Ka, i*L/(disc_int-1), L, I0))


    um_trap = (Iz[0]/((Iz[0] + Ks + (Iz[0]**2)/Ki)) + Iz[-1]/((Iz[-1] + Ks + (Iz[-1]**2)/Ki)))  
    km_trap = (Iz[0]/((Iz[0] + Ksl + (Iz[0]**2)/Kil)) + Iz[-1]/((Iz[-1] + Ksl + (Iz[-1]**2)/Kil))) 

    for i in range(1,disc_int-1):
        #print(i)
        um_trap     += 2*Iz[i]/(Iz[i] + Ks + (Iz[i]**2)/Ki)
        km_trap     += 2*Iz[i]/(Iz[i] + Ksl + (Iz[i]**2)/Kil)

    u_0 = u_m/20 * um_trap
    k_0 = k_m/20 * km_trap
        
    # variable rate equations
    dev_Cx  = u_0 * Cx * Cn/(Cn+K_N) - u_d*Cx
    dev_Cn  = - Y_nx * u_0 * Cx * Cn/(Cn+K_N) + Fnin 
    dev_Cl  = k_0 * Cn/(Cn+K_NL) * Cx - Kd * Cl * Cx

    ODEeq =  [dev_Cx, dev_Cn, dev_Cl]


    # Declare algebraic equations
    Aeq = []

    # Define objective to be minimized
    t           = SX.sym('t')

    return xd, xa, u, uncertainty, ODEeq, Aeq, states, algebraics, inputs, nd, na, nu, nmp



def integrator_model():
    """
    This function constructs the integrator to be suitable with casadi environment, for the equations of the model
    and the objective function with variable time step.
     inputs: NaN
     outputs: F: Function([x, u, dt]--> [xf, obj])
    """
    # load in desired variables
    _, _, env_dict              = specifications()
    mparams_names, mparams_vals = env_dict['v_names'], env_dict['v_vals']
    tf, movements               = env_dict['tf'], env_dict['steps']


    xd, xa, u, uncertainty, ODEeq, Aeq, \
        _, _, _, _, _, _, _ = DAE_system(mparams_names, mparams_vals)
    
    dt = tf/movements

    dae     = {'x': vertcat(xd), 'z': vertcat(xa), 'p': vertcat(u, uncertainty),
               'ode': vertcat(*ODEeq), 'alg': vertcat(*Aeq)}
    opts    = {'tf': dt}  # interval length
    F       = integrator('F', 'cvodes', dae, opts)

    return F