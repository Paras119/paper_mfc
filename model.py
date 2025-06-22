# import mesa
# from mesa.datacollection import DataCollector
# from numba import njit
# from numba.typed import Dict, List
# from numba.core import types
# import numpy as np
# from requirements import *
# from PreCode import *
# import tqdm
# import matplotlib.pyplot as plt


# class mfcModel(mesa.Model):

#     @staticmethod
#     def make_offsets(r=3):
#         offsets = []
#         rr = r*r
#         for dx in range(-r, r+1):
#             for dy in range(-r, r+1):
#                 for dz in range(-r, r+1):
#                     if dx == dy == dz == 0:    continue
#                     if dx*dx + dy*dy + dz*dz <= rr:    offsets.append((dx, dy, dz))
#         return np.array(offsets, dtype=np.int32)


#     def __init__(
#             self,
#             N=5,
#             substrate = 0, 
#             run_time=0,):
#         self.max_substrate = reactor_size ** 3
#         self.organism_number = N

#         # self.electrode_area = 2 * electrode_size ** 2
#         # self.electrode_center = reactor_size/2
#         # center - electode_size/2 < x < center - electode_size/2
#         # center - electode_size < y < center - electode_size
#         # z = 0
#         # if organisms are within x and y and z < 10 units, cells attach to biofilm


#         self.offsets = self.make_offsets(motility_radius)
#         self.reactor_size = reactor_size
#         self.occupancy = Dict.empty(key_type=types.int64, value_type=types.int8)
#         self.substrate = Dict.empty(key_type=types.int64, value_type=types.int8)
#         self.electrons = Dict.empty(key_type=types.int64, value_type=types.int8)
#         self.consumed_substrate_locations = Dict.empty(key_type=types.int64, value_type=types.int8)

        
#         self.consumed_substrate = 0
#         self.released_electrons = 0

#         self.pos_x = []
#         self.pos_y = []
#         self.pos_z = []
        
#         self.last_off = [0] * N
#         self.organism_growth_rate = [0] * N
#         self.organism_run_time = [0] * N

#         self.substrate_left = self.max_substrate

#         self.feed_count = [0] * N

#         # self.last_off = np.zeros(max_agents, dtype=np.int16)
#         # self.organism_growth_rate = np.zeros(max_agents, dtype=np.float32)
#         # self.organism_run_time = np.zeros(max_agents, dtype=np.int16)
#         # self.feed_count = np.zeros(max_agents, dtype=np.int16)


#         # # Set up data collection
#         model_reporters = {
#             'electrons released' : lambda m: m.released_electrons/time_step,
#             'biomass produced' : lambda m: m.organism_number,
#             'substrate left' : lambda m: m.substrate_left,
#             'Organism x location' : lambda m: m.pos_x,
#             'Organism y location' : lambda m: m.pos_y,
#             'Organism z location' : lambda m: m.pos_z
#         }

#         self.datacollector = DataCollector(model_reporters)

#         # Initialize agents in 3D grid
#         for i in range(N):
#             vx = int(organism_initial_base_pos[i][0])
#             vy = int(organism_initial_base_pos[i][1])
#             vz = int(organism_initial_vertical_pos[i])
#             self.pos_x.append(vx)
#             self.pos_y.append(vy)
#             self.pos_z.append(vz)

#             self.organism_growth_rate[i] = growth_rate_distribution[i]
            
#             idx = compute_bash_loc(vx, vy, vz, reactor_size)
#             self.occupancy[idx] = 1
#             self.substrate[idx] = initial_substrate_per_voxel  # Start with substrate at agent positions

#         self.running = True
#         self.datacollector.collect(self)


#     def step(self):

#         # 1) find_loc_and_move_cells
#         if len(self.pos_x) < self.organism_number * 10:
#             self.consumed_substrate_locations.clear()
#             self.substrate_left = self.max_substrate
#             self.consumed_substrate = 0

#         self.pos_x, self.pos_y, self.pos_z, self.occupancy, self.substrate, self.feed_count, self.consumed_substrate, self.released_electrons, self.consumed_substrate_locations = find_loc_and_move_cells(self.pos_x, self.pos_y, self.pos_z, self.last_off, self.occupancy, self.substrate, self.offsets, self.reactor_size, self.feed_count, self.consumed_substrate, self.released_electrons, self.consumed_substrate_locations)

#         # 2) Python pass: handle reproduction & death
#         self.organism_run_time, self.organism_growth_rate, self.offsets, self.feed_count, self.last_off, self.occupancy, self.substrate, self.pos_x, self.pos_y, self.pos_z, self.organism_number = births_and_deaths(self.organism_run_time, self.organism_growth_rate, self.offsets, self.feed_count, self.last_off, self.occupancy, self.substrate, self.organism_number, self.pos_x, self.pos_y, self.pos_z)

#         self.substrate_left = self.max_substrate - self.consumed_substrate
        

#         # 3) Data collection
#         self.datacollector.collect(self)

        

# model = mfcModel(N = N)
# x = []
# s = []
# e = []
# for i in tqdm.tqdm(range(100)):
#     model.step()
#     print(' organism number ', model.organism_number)
#     print(' substrate left ', model.substrate_left)
#     print('consumed locations ', len(model.consumed_substrate_locations))
#     x.append(model.organism_number)
#     s.append(model.substrate_left)
#     e.append(model.released_electrons * 1e9/coulomb * external_resistor/time_step)

# plt.plot(x, label='biomass')
# plt.plot(s, label='substrate')
# plt.legend()
# plt.show()


# # plt.plot(e, label='electrons')
# # plt.legend()
# # plt.show()


import mesa
from mesa.datacollection import DataCollector
from numba import njit
from numba.typed import Dict, List
from numba.core import types
import numpy as np
from requirements import *
from PreCode import *
import tqdm
import matplotlib.pyplot as plt


class mfcModel(mesa.Model):

    @staticmethod
    def make_offsets(r=3):
        offsets = []
        rr = r*r
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                for dz in range(-r, r+1):
                    if dx == dy == dz == 0:    continue
                    if dx*dx + dy*dy + dz*dz <= rr:    offsets.append((dx, dy, dz))
        return np.array(offsets, dtype=np.int32)


    def __init__(
            self,
            N=5,
            substrate = 0, 
            run_time=0,):
        self.max_substrate = reactor_size ** 3
        self.organism_number = N

        # self.electrode_area = 2 * electrode_size ** 2
        # self.electrode_center = reactor_size/2
        # center - electode_size/2 < x < center - electode_size/2
        # center - electode_size < y < center - electode_size
        # z = 0
        # if organisms are within x and y and z < 10 units, cells attach to biofilm


        self.offsets = self.make_offsets(motility_radius)
        self.reactor_size = reactor_size
        self.occupancy = Dict.empty(key_type=types.int64, value_type=types.int8)
        self.substrate = Dict.empty(key_type=types.int64, value_type=types.int8)
        self.electrons = Dict.empty(key_type=types.int64, value_type=types.int8)
        self.consumed_substrate_locations = Dict.empty(key_type=types.int64, value_type=types.int8)

        
        self.consumed_substrate = 0
        self.released_electrons = 0

        self.pos_x = []
        self.pos_y = []
        self.pos_z = []
        
        self.last_off = [0] * N
        self.organism_growth_rate = [0] * N
        self.organism_run_time = [0] * N
        self.feed_count = [0] * N

        # self.last_off = np.zeros(max_agents, dtype=np.int16)
        # self.organism_growth_rate = np.zeros(max_agents, dtype=np.float32)
        # self.organism_run_time = np.zeros(max_agents, dtype=np.int16)
        # self.feed_count = np.zeros(max_agents, dtype=np.int16)


        # # Set up data collection
        model_reporters = {
            'electrons_released_per_time_step' : lambda m: m.released_electrons/time_step,
        #     "Planktonic Cells": lambda m: len(m.agents),
        #     "Substrate": lambda m: m.substrate,
        #     "Attached Cells": lambda m: len([a for a in m.agents if a.attached]),
        #     "Planktonic cells": lambda m: [list(a.pos).append(a.z) for a in m.agents if not a.attached and a.pos],
        #     "Attached cell locations": lambda m: [a.pos for a in m.agents if a.attached and a.pos],
        }

        self.datacollector = DataCollector(model_reporters)

        # Initialize agents in 3D grid
        for i in range(N):
            vx = int(organism_initial_base_pos[i][0])
            vy = int(organism_initial_base_pos[i][1])
            vz = int(organism_initial_vertical_pos[i])
            self.pos_x.append(vx)
            self.pos_y.append(vy)
            self.pos_z.append(vz)

            self.organism_growth_rate[i] = growth_rate_distribution[i]
            
            idx = compute_bash_loc(vx, vy, vz, reactor_size)
            self.occupancy[idx] = 1
            self.substrate[idx] = initial_substrate_per_voxel  # Start with substrate at agent positions

        self.running = True
        self.datacollector.collect(self)


    def step(self):

        # 1) find_loc_and_move_cells
        if len(self.pos_x) < 100:
            self.consumed_substrate_locations.clear()
            self.substrate_left = self.max_substrate
            self.consumed_substrate = 0

        self.pos_x, self.pos_y, self.pos_z, self.occupancy, self.substrate, self.feed_count, self.consumed_substrate, self.released_electrons, self.consumed_substrate_locations = find_loc_and_move_cells(self.pos_x, self.pos_y, self.pos_z, self.last_off, self.occupancy, self.substrate, self.offsets, self.reactor_size, self.feed_count, self.consumed_substrate, self.released_electrons, self.consumed_substrate_locations)

        # 2) Python pass: handle reproduction & death
        self.organism_run_time, self.organism_growth_rate, self.offsets, self.feed_count, self.last_off, self.occupancy, self.substrate, self.pos_x, self.pos_y, self.pos_z, self.organism_number = births_and_deaths(self.organism_run_time, self.organism_growth_rate, self.offsets, self.feed_count, self.last_off, self.occupancy, self.substrate, self.organism_number, self.pos_x, self.pos_y, self.pos_z)

        self.substrate_left = self.max_substrate - self.consumed_substrate
        

        # 3) Data collection
        self.datacollector.collect(self)

        

model = mfcModel(N = N)
x = []
s = []
e = []
for i in tqdm.tqdm(range(100)):
    model.step()
    # print(' organism number ', model.organism_number)
    # print(' substrate left ', model.substrate_left)
    print(model.released_electrons)
    x.append(model.organism_number)
    s.append(model.substrate_left)
    e.append(model.released_electrons * 1e9/coulomb * external_resistor/time_step)

plt.plot(x, label='biomass')
plt.plot(s, label='substrate')
plt.legend()
plt.show()


plt.plot(e, label='electrons')
plt.legend()
plt.show()