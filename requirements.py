import numpy as np 

# 10000
reactor_size = 1000  # every unit voxel corresponds to around 1e9 substrate molecules, hence 1e8 * 1e12 = 1e20 = 1mM
# Now electrons released per substrate molecule let be 8, with coulombic efficiency 100% 
# per unit substrate electron released = 8 units/voxel per organism

electrode_size = 0.01 * reactor_size
N = 10000 # max achievable is 1e6, but can handle rather slowly till 1e7 not beyond
max_agents = 1e6
# 10000000
motility_radius = 1

per_cell_electrons = 8
coulombic_efficiency = 1
external_resistor = 4600
coulomb = 6.241e18 # electrons
single_voxel_representation = 1e8

time_step = 60*60*2

death_rate_percentage = 1/0.4

r = 3

growth_rate_distribution = np.random.normal(np.log(2) / 10, np.log(2) / 10 * 0.1, N)

initial_substrate_per_voxel = 1

organism_initial_base_pos = np.random.uniform(0, reactor_size, (N, 2))
organism_initial_vertical_pos = np.random.uniform(0, reactor_size, N)