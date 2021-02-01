# MC-rollouts
Code base for MC sampling of a process model

- Define process model in casadi via Dynamic_system.py
- Define parametric uncertainty in Dynamic_system.py
- Define other miscellaneous parameters e.g. upper and lower bound on controls in Dynamic_system.py
- MPC controller is detailed by direct_collocation_lutein_1FE_sim.py
- ValidationExperiment.py is a class for performing MC rollouts it integrates the controller and dynamic system
- stores data in a h5py file, and then saves at the path detailed by user - more info on h5py files and how to unpack data
  for further analysis here - https://docs.h5py.org/en/stable/
  
  
