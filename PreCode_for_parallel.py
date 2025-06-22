import numpy as np
import pandas as pd
from numba import njit, int32, int64, prange
from numba.typed import Dict, List
from numba.core import types
from requirements import *

@njit
def compute_index(x, y, z, size):
    return (x * size + y) * size + z

@njit(parallel=True)
def step_move_and_eat(
    pos_x, pos_y, pos_z,
    last_off,
    is_alive,
    occupancy,
    substrate,
    agent_count,
    feed_count,
    consumed_substrate_arr,
    electrons_arr
):
    temp_cons = 0
    temp_elec = 0.0
    size = reactor_size
    for i in prange(agent_count):
        if is_alive[i] == 0:
            continue
        x = pos_x[i]; y = pos_y[i]; z = pos_z[i]
        base = compute_index(x, y, z, size)
        start = last_off[i] % MOFF
        for k in range(MOFF):
            d = (start + k) % MOFF
            dx, dy, dz = OFFSETS[d]
            nx, ny, nz = x+dx, y+dy, z+dz
            if nx < 0 or nx >= size or ny < 0 or ny >= size or nz < 0 or nz >= size:
                continue
            idx2 = (nx*size + ny)*size + nz
            if occupancy[idx2] == 0 and substrate[idx2] > 0:
                # move
                occupancy[base]  = 0
                occupancy[idx2] = 1
                pos_x[i], pos_y[i], pos_z[i] = nx, ny, nz
                last_off[i] = d + 1
                # eat
                substrate[idx2] -= 1
                temp_cons += 1
                temp_elec += PER_CELL_ELECTRONS
                feed_count[i] += 1
                break
    consumed_substrate_arr[0] = temp_cons
    electrons_arr[0] = temp_elec * COULOMBIC_EFFICIENCY