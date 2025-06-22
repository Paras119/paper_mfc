import mesa
from mesa.datacollection import DataCollector
from numba import njit
from numba.typed import Dict, List
from numba.core import types
import numpy as np
from requirements import *
from PreCode_for_parallel import *
import tqdm
import matplotlib.pyplot as plt


class mfcModel(mesa.Model):
    def __init__(self, N=500, run_time=0):
        self.reactor_size = reactor_size
        self.NVOX = reactor_size**3

        # --- Fixed buffers ---
        MAX_AGENTS = max_agents
        self.agent_count = N
        self.pos_x      = np.zeros(MAX_AGENTS, dtype=np.int32)
        self.pos_y      = np.zeros(MAX_AGENTS, dtype=np.int32)
        self.pos_z      = np.zeros(MAX_AGENTS, dtype=np.int32)
        self.last_off   = np.zeros(MAX_AGENTS, dtype=np.int32)
        self.feed_count = np.zeros(MAX_AGENTS, dtype=np.int32)
        self.growth_rate= np.zeros(MAX_AGENTS, dtype=np.float32)
        self.run_time   = np.zeros(MAX_AGENTS, dtype=np.float32)
        self.is_alive   = np.zeros(MAX_AGENTS, dtype=np.int8)

        # Pre-allocate 1D grids
        self.occupancy = np.zeros(self.NVOX, dtype=np.int8)
        self.substrate = np.full(self.NVOX, initial_substrate_per_voxel, dtype=np.int16)
        self.electrons = np.zeros(self.NVOX, dtype=np.float32)  # if you need spatial track


        # fill first N agents
        for i in range(N):
            x,y,z = organism_initial_base_pos[i][0], organism_initial_base_pos[i][1], organism_initial_vertical_pos[i]
            self.pos_x[i], self.pos_y[i], self.pos_z[i] = x,y,z
            self.is_alive[i] = 1
            self.growth_rate[i] = growth_rate_distribution[i]

        consumed_arr = np.zeros(1, np.int32)
        elec_arr     = np.zeros(1, np.float64)

        # inside step():
        self.pos_x, self.pos_y, self.pos_z, \
        self.occupancy, self.substrate, \
        self.feed_count, consumed_arr, elec_arr = \
            step_move_and_eat(
                self.pos_x, self.pos_y, self.pos_z,
                self.last_off,
                self.occupancy,
                self.substrate,
                self.agent_count,
                self.feed_count,
                consumed_arr,
                elec_arr
            )
        self.consumed_substrate = consumed_arr[0]
        self.released_electrons = elec_arr[0]

        # Mark deaths & collect new daughter positions into pos_*[agent_count...]
        # (Use simple loops or a second njit kernel that atomically bumps agent_count.)
        # Then, serially compact:
        write=0
        for read in range(self.agent_count):
            if self.is_alive[read]:
                if write != read:
                    # copy state
                    for arr in (self.pos_x, self.pos_y, self.pos_z,
                                self.last_off, self.feed_count,
                                self.growth_rate, self.run_time, self.is_alive):
                        arr[write] = arr[read]
                write += 1
        self.agent_count = write



