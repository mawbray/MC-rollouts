import numpy as np
import csv
import pandas as pd
import torch
import timeit
#from MLPClass import Net as PolicyNet
#from torch.distributions import MultivariateNormal as MNorm
#from torch.distributions import Normal as Norm
#from RunningStandard import RunningStat as Zfilter
import math
import matplotlib.lines as Lines
import h5py
from datetime import datetime
import scipy.integrate as scp
import matplotlib.pyplot as plt
import numpy.random as rnd
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
eps  = np.finfo(float).eps
from casadi import *
from datetime import datetime
import datetime as DT
import seaborn as sns
#from episode_simulation import *
#from PPO import PPOAgent
#from SSE import SSE_MIMO
from csvkit import *
from Dynamic_system import *
import os
from scipy.stats import beta as beta
from tqdm import tqdm
from direct_collocation_lutein_1FE_sim import Lutein_NMPC as NMPC_controller


class Validation_Experiment(object):
    def __init__(self, **kwargs):
        ################################################
        #### ------- Initialisation Methods ------- ####
        ################################################
        # importing bioporcess model, defined via CasADI, imported via Dynamic System
        self.info_integration   =  [specifications, DAE_system, integrator_model ]
        _, _, self.env_dict     =  specifications()
        

        # defining containers and other hyperparameters associated with simulation 
        self.kwargs         = kwargs
        self.iter           = kwargs['optitbackoffs']
        self.savepath       = kwargs['savepath']
        self.path1          = kwargs['path1']
        self.nx, self.nu    = self.env_dict['nx'], self.env_dict['nu']
        self.MC             = kwargs['MC']    
        self.N              = self.env_dict['steps']
        self.tf             = self.env_dict['tf']
        self.sigma_p        = self.env_dict['p_sig']
        self.dt             = self.tf/self.N
        self.weights        = kwargs['weights']
        self.BO_part        = kwargs['BO']

        # defining pytorch regulars
        # code regularisation
        self.dtype          = torch.float
        self.use_cuda       = torch.cuda.is_available()
        self.device         = torch.device("cpu")      # cuda:0" if self.use_cuda else 
        torch.cuda.empty_cache() 

        # policy and environment initialisation
        self.load_agent_params()
        self.construct_agent()
        self.load_env_params()
        self.init_containers()

        # miscellaneous initialisations 
        # Running Standardisation 
        #self.standardise = Zfilter(self.nx, demean=True, destd=True, clip=10)
        #self.standardise.load(self.kwargs['path1'], self.kwargs['optitbackoffs'])
        # initialising h5py file for dataset storage and collation
        # this is very lazy, I should define the path as a variable, but these things happen
        self.date        = datetime.date(datetime.today())           # + DT.timedelta(days=1)
        if os.path.isdir(os.path.join(self.savepath, 'Validation')) != True:
            os.mkdir(os.path.join(self.savepath, 'Validation')) 
        self.f  = h5py.File(os.path.join(self.savepath, f'Validation\\validation_transition_store_{self.iter}_{self.BO_part}.hdf5'),'w')
        

    def init_containers(self):

        nu, nx, N, MC  = self.nu, self.nx, self.N, self.MC
        self.his_z, self.his_y        = np.zeros([nu+nx,N,MC]), np.zeros([nx,N,MC])
        self.his_g, self.his_Gt       = np.zeros([nu+nx,N,MC]), np.zeros([nx,N,MC])

    def load_agent_params(self):

        path1   = self.kwargs['path1']
        path    = os.path.join(path1,'agent.csv')


        x       = pd.read_csv(path, header = 0)
        values  = x.to_numpy()
        columns = x.columns

        


        self.policydict = {columns[i]: values[0,i] for i in range(len(columns))}

        return 

    def load_env_params(self):

        path1   = self.kwargs['path1']
        path    = os.path.join(path1,'env.csv')
        x       = pd.read_csv(path, header = 0)
        values  = x.to_numpy()
        columns = x.columns

        self.envsets = {columns[i]: values[0,i] for i in range(len(columns))}

    def construct_agent(self):
        
        args    = self.policydict
        iter    = self.kwargs['optitbackoffs']
        path1   = self.kwargs['path1']
        self.UB, self.LB    = np.array(self.env_dict['absmax']), np.array(self.env_dict['absmin'])

        # defining NMPC agent 

        policy_kwargs   = {'order': 5, 'points': 'radau'}
        self.policy     = NMPC_controller(**policy_kwargs)
        
        return

    ################################################
    ###### ------- Simulation Methods ------- ######
    ################################################


    def run_ep(self):
        # method for collecting one episode of experience

        # setting setpoint of Agent and getting key args
                          
        argsA       = self.policydict
        nu          = argsA['output_size']
        movements   = argsA['steps']
        U_UB        = argsA['U_UB']
        U_LB        = argsA['U_LB']
        gamma       = argsA['gamma']
        tau         = argsA['tau']
        kappa       = argsA['kappa']


        # creating memory 
        xt              = []
        tt              = []
        c_hist          = []
        R               = []
        fin             = []
        violations      = []
        cons_pen        = []

        # initialising environment and state
        state, cons_init = self.reset()
        ns    = len(state)
        s     = 0
    
        xt.append(state)
        violations.append(cons_init)
        tt.append(s)

        # simulation of trajectory
        while True:
            prev_state          = state
            # selecting action given the state of the environment
            action              = self.act(np.array(state).reshape(1,-1), s)    # select control for this step from that possible,
            # stepping the environment given current state and action, returns state, reward, done?, info and violation
            state, r, done, info, vio  = self.step(action)
            # storing transition in memory store

            s += 1
        
            R.append(r)
            xt.append(state)
            tt.append(s)
            c_hist.append(action.reshape(1,nu))
            fin.append(done)
            violations.append(vio)
           
            if done == 0:
              break

        # data processing
        states, thyme, controls, rewards, done, violate = \
            np.array(xt), np.array(tt), np.array(c_hist),\
            np.array(R), np.array(fin), np.array(violations)
        
        controls = controls.squeeze(axis =1) 

        # outputs
        Treturn     = self.discount(rewards, done, argsA)

        # this is required for plotting purposes (appends last control again, for step plot)
        cont            = np.zeros((controls.shape[0]+1, controls.shape[1]))
        x               = controls[-1]
        cont[:-1]       = controls
        cont[-1]        = x
        controls        = cont
 

        return Treturn, states, thyme, controls, violations

    def discount(self, rewards,  masks, argsA):
    

        returns         = np.zeros((rewards.shape))
        gamma           = argsA['gamma']
        kappa           = argsA['kappa']
        prev_value      = 0.
    
        for i in reversed(range(rewards.shape[0])):
            # calculating undiscounted rewards i.e. objective
       
            returns[i] =  rewards[i]  +  prev_value * masks[i] 
            prev_value  = returns[i]

        returns = returns 

        return returns[0]

    def sample_eps(self):
        # Simulation takes environment, imparts control action from stochastic policy and simulates, observes next state to the end of the sequence and outputs reward
        # internal definitions
        argsA                       = self.policydict
        nu                          = argsA['output_size']
        steps                       = argsA['steps']
        ng                          = argsA['n_cons']

        # compile state and control trajectories
        valid_traj      = self.kwargs['MC']


        # run model validation 
   
        rv          = []
        rv_m        = []
        actions     = []
        states      = []
        thyme       = []
        print('validation')
        violations  = []
        violate     = np.zeros((steps+1, ng, valid_traj))

        for x in range(valid_traj):
            # drawing initial state and setpoints
            
            Gt, x_state, t_t, c__hist, vio = self.run_ep()
            rv.append(Gt)
            states.append(x_state)
            actions.append(c__hist)
            thyme.append(t_t)
            violations.append(vio)

            self.his_z[:self.nx,:,x], self.his_z[self.nx:,:,x] = x_state[:-1,:].T, c__hist[:-1,:].T
            self.his_y[:,:,x]   = x_state[1:,:].T

        # store mean reward and state expectation from trajectory  
        r_mean, r_std = np.mean(rv, axis=0), np.std(rv, axis =0)
        rv_m.append([r_mean, r_std])

        ############################################################################
        # --- analysing violations and quantifying probability of satisfaction --- #    
        ############################################################################           
        count = 0.
        for i in range(len(violations)):
            x   = np.array(violations[i])
            violate[:,:,i] = x
            if ( x > 0 ).any():
                count += 1

        F_vioSA     = count/valid_traj
        alpha       = valid_traj + 1 - valid_traj * F_vioSA
        b_ta        = valid_traj * F_vioSA + 1e-8
        conf        = self.policydict['confidence']
        betaDist    = beta(alpha, b_ta)
        F_LB        = betaDist.ppf(conf)

        print('probability of constraint satisfaction = ', F_LB)
            
        # plot check
        
        print('plot please')
        # settings
        font = {'family' : 'serif',
        'weight' : 'bold','size'   : 16}

        plt.rc('font', **font)  # pass in the font dict as kwargs
        plt.rc('axes', titlesize=18)        # fontsize of the axes title
        plt.rc('axes', labelsize=16)        # fontsize of the x and y label 
        date        = datetime.date(datetime.today())           # + DT.timedelta(days=1)

        # action color map 
        Acmap = sns.color_palette("Blues", n_colors=1)
        Ecmap = sns.color_palette("Greens", n_colors=1)

        # state color map
        Acmap1 = sns.color_palette("plasma", n_colors=1)
        Ecmap1 = sns.color_palette("viridis", n_colors=1)
        Stepmap = sns.color_palette("OrRd", n_colors=1)
        print(np.array(actions).shape)
        # dataframe prep
        df_x        = pd.DataFrame({'X': np.concatenate(states)[:,0]}, index = np.concatenate(thyme), columns=["X"])
        df_n        = pd.DataFrame({'N': np.concatenate(states)[:,1]}, index = np.concatenate(thyme), columns=["N"])
        df_L        = pd.DataFrame({'L': np.concatenate(states)[:,2]}, index = np.concatenate(thyme), columns=["L"])
        df_control  = pd.DataFrame({'Fn': np.concatenate(actions)[:,0] }, index = np.concatenate(thyme), columns=["Fn"])
        df_controL  = pd.DataFrame({'I': np.concatenate(actions)[:,1]}, index = np.concatenate(thyme), columns=["I"])
            
        # finding mean and standard deviation of control response
        cont_in     = np.zeros((steps+1, nu, valid_traj))
        cont        = np.concatenate(actions)
        cont[:,0]   = cont[:,0]      # converting nitrate from ml/h to mg/h based on NaNO3 MR
        ep_index    = np.linspace(0, steps, steps+1)
        for i in range(valid_traj):
            cont_in[:,:,i]  = cont[i*(steps+1):(i+1)*(steps+1),:] 

        std_cont = np.std(cont_in, axis =2)
        mu_cont  = np.mean(cont_in, axis =2)
                                                          
        # plot distribution of actions and states
        # biomass plot
        fig     = plt.figure(figsize = (50,20))
        ax      = plt.subplot(5,1,1)
        g       = sns.lineplot(data = df_x, palette = Acmap1, ci = 'sd',  linewidth= 4)
        g.lines[-2].set_linestyle("--")
        plt.xlabel('Control Interaction', labelpad=20)
        plt.ylabel(r'Biomass Concentration $g L^{-1}$', labelpad=20)
        handles, labels = ax.get_legend_handles_labels()
        labels = ['X']
        ax.legend(handles=handles, labels=labels)
        ax.tick_params(which='major', length=12, width =2.5)
        ax.tick_params(which='minor', length=8, width =2.5)
        # nitrate plot
        ax      = plt.subplot(5,1,2)
        g       = sns.lineplot(data = df_n, palette = Ecmap1, ci = 'sd', linewidth= 4)
        g.lines[-2].set_linestyle("--")
        plt.xlabel('Control Interation', labelpad=20)
        plt.ylabel(r'Nitrate Concentration  $mg L^{-1}$', labelpad=20)
        handles, labels = ax.get_legend_handles_labels()
        labels = ['N']
        ax.legend(handles=handles, labels=labels)
        ax.tick_params(which='major', length=12, width =2.5)
        ax.tick_params(which='minor', length=8, width =2.5)
        # lutein plot
        ax      = plt.subplot(5,1,3)
        g       = sns.lineplot(data = df_L, palette = Acmap, ci = 'sd', linewidth=4)
        g.lines[-2].set_linestyle("--")
        plt.xlabel('Control Interaction', labelpad=20)
        plt.ylabel(r'Lutein Concentration $mg L^{-1}$', labelpad=20)
        handles, labels = ax.get_legend_handles_labels()
        labels = ['U']
        ax.legend(handles=handles, labels=labels)
        ax.tick_params(which='major', length=12, width =2.5)
        ax.tick_params(which='minor', length=8, width =2.5)
        # control plot
        ax      = plt.subplot(5,1,4)
        plt.plot(ep_index, mu_cont[:,0], 'k', mew=2, drawstyle = 'steps-post', label ='F_in', linewidth= 4 )
        plt.gca().fill_between(ep_index.flat, 
                    mu_cont[:,0] - 1*std_cont[:,0], 
                    mu_cont[:,0] + std_cont[:,0], step = 'post',
                    color='c0', alpha=0.2, label = f'GP conf interval {i}')
        #g       = sns.lineplot(data = df_Nin, palette = Acmap, drawstyle='steps-post', ci = 'sd', linewidth=2)
        #g.lines[0].set_linestyle("--")
        plt.xlabel('Control Interaction', labelpad=20)
        plt.ylabel(r'Nitrate Inflow $mg h^{-1}$', labelpad=20)
        handles, labels = ax.get_legend_handles_labels()
        labels = ['N_in']
        ax.legend(handles=handles, labels=labels)
        ax.tick_params(which='major', length=12, width =2.5)
        ax.tick_params(which='minor', length=8, width =2.5)
        ax      = plt.subplot(5,1,5)
        plt.plot(ep_index, mu_cont[:,1], 'k', mew=2, drawstyle = 'steps-post', label ='F_in', linewidth= 4 )
        plt.gca().fill_between(ep_index.flat, 
                    mu_cont[:,1] - 1*std_cont[:,1], 
                    mu_cont[:,1] + std_cont[:,1], step = 'post',
                    color='c0', alpha=0.2, label = f'GP conf interval {i}')
        #g       = sns.lineplot(data = df_Nin, palette = Acmap, drawstyle='steps-post', ci = 'sd', linewidth=2)
        #g.lines[0].set_linestyle("--")
        plt.xlabel('Control Interaction', labelpad=20)
        plt.ylabel(r'Light Intensity $\mu E$', labelpad=20)
        handles, labels = ax.get_legend_handles_labels()
        labels = ['N_in']
        ax.legend(handles=handles, labels=labels)
        ax.tick_params(which='major', length=12, width =2.5)
        ax.tick_params(which='minor', length=8, width =2.5)

        plt.subplots_adjust(wspace = .2)
        plt.subplots_adjust(hspace = .15)
        if os.path.isdir(os.path.join(self.savepath, 'Validation')) != True:
            os.mkdir(os.path.join(self.savepath, 'Validation')) 
        plt.savefig(os.path.join(self.savepath, f'Validation\\Validation_Rollouts_{self.BO_part}.svg'))
        plt.close()

     
          
        # store experience from epoch 
        self.StoreEpisode(states, thyme, actions, rv, violate, self.iter)
        self.saves(F_LB, rv_m )

                     
    
        # -- print progress -- #
        #print('epoch:', epoch)
        #print(f'mean objful: {obj_mean} +- {obj_std}')
        print(f'mean reward: {r_mean} +- {r_std}')

        return r_mean, r_std, F_LB, self.his_z, self.his_y

    ################################################
    ##### ------- Integration & Models ------- #####
    ################################################

    def step(self, action):
        # class method for integration of environment (i.e. environment transition)
        # allocating states (current state and for tracking progress)
        state           = self.current_state
        prev_state      = state
        s               = self.s
    

        # step environment one time step and return observation and reward given action   
        current_state   = self.integration(state, action) 
        self.current_state = [float(current_state[i]) for i in range(current_state.shape[0])]
        #current_state   = self.measurement_noise(current_state)
        transition      = current_state - prev_state
        current_state   = [float(current_state[i]) for i in range(current_state.shape[0])]
        reward          = self.rewardfunc(current_state, prev_state, action)

        # incremeting step counter and time
        self.time       += self.dt
        self.s          += 1
        ##################################
        # assess constraint violation here
        ##################################
        violation = self.constraint(current_state, action)
       
        if self.s == self.N:
          done      = 0
          info      = {'state_feature': self.s, 'time': self.time}
        else: 
          done  = 1
          info      = {'state_feature': self.s, 'time': self.time}

        return current_state, reward, done, info, violation

    def integration(self,current_state, action):
        
        # Performing system integration
        F = self.info_integration[-1]()
        Sigma_p = self.Sigma_p
        # Performing integration 
        xd = F(x0=vertcat(np.array(current_state)), p=vertcat(action, Sigma_p))
        integrated_state = np.array(xd['xf'].T)[0] #+np.random.multivariate_normal([0.]*self.nx,np.array(Sigma_v))
        x = integrated_state #+np.random.multivariate_normal([0.]*self.nx,np.array(Sigma_v)*0.1)
        # Physical Bounds
        for ii in range(self.nx):
            if integrated_state[ii] < 0:
                integrated_state[ii] = 0
        for ii in range(self.nx):
            if x[ii] < 0:
                x[ii] = 0
       
        return x.reshape((self.nx,1))

    def reset(self):
        # method for resetting the initial state at t=0
        self.construct_agent()
        x0                  = self.env_dict['x0']
        self.current_state  = np.array([float(np.random.normal(x0[0], x0[0]*0.0125)), float(np.random.normal(x0[1], x0[1]*0.0125)), float(np.random.normal(x0[2], 0))])
        self.s              = 0
        self.time           = 0.
        self.VTin           = 0. # constraint on volume added
        self.prev_u         = np.zeros((1,self.nu))
        # Declaring noisy disturbance
        Sigma_v = self.env_dict['sigma_noisev'] * diag(np.ones(self.nx)) * 1e-6
        self.Sigma_p = np.random.multivariate_normal([0.]*len(self.sigma_p), self.sigma_p)

        cons_init           = self.constraint(self.current_state, np.zeros((1,2)))

        return self.current_state, cons_init

    ################################################
    ###### ------- Allocation Methods ------- ######
    ################################################


    def rewardfunc(self, state, prev_state, control):
        # based on linear function of basis features
        # RL specific
        transition = np.array(state) - np.array(prev_state)
        trans_u    = np.absolute(control - self.prev_u).reshape(1,-1)*np.array([1/10, 1/1000]).reshape(1,-1)
        state      = state * np.array([1, 1/100, 1])

        if self.s ==  self.N - 1:
            reward  =   np.matmul(self.weights, state) - (4 * trans_u[0,0])**2 - (9 * trans_u[0,1])**2                      # - 0.1*control[0] - 0.5*control[1]/1000  - control[0]/50 
        else:
            reward = - (4 * trans_u[0,0])**2 - (9 * trans_u[0,1])**2                                                       # - 0.1*control[0] - 0.5* control[1]/1000  - control[0]/50 


        self.prev_u = control                                                 # - 0.1*control[0] - 0.5* control[1]/1000


        
        
        #print(reward)

        # internal definitions
        """
        time_2_term =   self.steps - self.s
        reward      =   self.args['gamma1']** time_2_term * reward  
        """
        
        #print(reward)
        
        return reward


    def constraint(self, current_state, control):
       # assessing constraint violation under scenario
       # control specific
       envsets  = self.envsets
       b    = self.kwargs['b']
       A    = self.kwargs['A']
       ng   = envsets['ng']

       
       cons     = [None for _ in range(ng)]
       

       current_state = current_state * np.array([1, 1e-3, 1e-3])            # scaling to grams

       
       for i in range(0,ng):
           #print(A.shape, i )
           x        = A[i,:] @ current_state - b[i]
           cons[i]  = x


       """# soft chance constraints 
       cons2    = max(0, Q/(2.1*X) - 1)
       cons3    = max(0, 100/N - 1)
       
       cons     = [cons1, cons2, cons3]"""
       #print(cons)
             
       return cons

    ################################################
    ###### ------- Poli. Act. Methods ------- ######
    ################################################
    # RL specifc
    def hiDprojection(self, state):

        X, N, Q = state[:,0], state[:,1], state[:,2]
        output = torch.zeros((1,X.shape[0], self.policydict['obs_size']), dtype = torch.float64).to(self.device)
        output[0,:,0], output[0,:,1], output[0,:,2],  = X, N, Q   #output[:,0,2] output[0,:,3], output[0,:,4]

        return output

    def act(self, state, s):
        # choose your control at t=s here
        # variable definitions 
        steps           = self.N
        time_to_go      = self.tf - self.time
        cont_int_2go    = steps - self.s 

        # normalising raw state based on data distribution in memory
        action = self.policy.forecast(state.squeeze(), time_to_go, cont_int_2go)
        action = action.reshape(1,-1)

        # enforcing bounds on action space
        UB, LB  = self.UB, self.LB
        for i in range(action.shape[1]):
            if action[0,i] < LB[i]:
                #print(True)
                action[0,i] = LB[i]
            if action[0,i] > UB[i]:
                action[0,i] = UB[i]

        action  = action.squeeze()
        #print(action)
 
        return action


    ################################################
    ##### ------- Storage etc. Methods ------- #####
    ################################################


    def StoreEpisode(self,states, time, actions, train_rwd, violate, i, **kwargs):
        # create datafile for each epoch
     
        #states, thyme, actions, rv, epoch, iteration
        grp     = self.f.create_group(f"trajectories_Agent_0_{i}_{self.BO_part}")
        subgrp1 = grp.create_group('states')
        ds1     = subgrp1.create_dataset('state_traj', data=np.array(states))
        subgrp2 = grp.create_group('actions')
        ds2     = subgrp2.create_dataset('action_traj', data=np.array(actions).squeeze())
        subgrp4 = grp.create_group('rwd_train')
        ds4     = subgrp4.create_dataset('rwd_tran', data=np.array(train_rwd).squeeze())
        subgrp5 = grp.create_group('time')
        ds5     = subgrp5.create_dataset('time_traj', data=np.array(time).squeeze())
        subgrp6 = grp.create_group('violations')
        ds6     = subgrp5.create_dataset('violate_traj', data=np.array(violate).squeeze())

        return

    def saves(self, F_LB, returns):
        
        path1       = os.path.join(self.savepath, f'Validation')
        csvwriter(os.path.join(path1,f'agent_{self.BO_part}.csv'), self.policydict)
        csvwriter(os.path.join(path1,f'experiment_{self.BO_part}.csv'), self.env_dict)
        csvwriter(os.path.join(path1,f'environment_{self.BO_part}.csv'), self.envsets)
        csvwriter(os.path.join(path1,f'real_chance_satisfaction_{self.iter}_{self.BO_part}.csv'), {'Real Satisfaction': F_LB})
        csvwriter(os.path.join(path1,f'performance_{self.iter}_{self.BO_part}.csv'), {'Real Performance': returns})

        return


b   = np.array([2.6, -0.15, 0])
A   = np.array([[1, 0, 0], [0, -1, 0], [-1.67/1000, 0, 1]])
path1       = "C:\\Users\\g41606mm\\Dropbox\\Projects\\Python\\RL\\CCPO w Uncertainty\\PPO2\\LuteinCS\\GPSS_simple_2\\2021-01-17\\mission_4"
savepath    = "C:\\Users\\g41606mm\\Dropbox\\Projects\\Python\\MPC\\CCPOProject\\LuteinCS\\Validation"
#path1       = savepath 
opt_iter    = 0
weights     = np.array([0, -1e-1, 4])

Vkwargs ={'path1': path1, 'A': A, 'b': b, 'savepath': savepath,
        'optitbackoffs': opt_iter  , 'MC': 500, 'weights': weights, 'BO': 'MPC'}


Vexp = Validation_Experiment(**Vkwargs)
_,_,_, _,_  = Vexp.sample_eps() 




        
        



        






