# import numpy as np
# import pandas as pd
# from numba import njit, int32, int64
# from numba.typed import Dict, List
# from numba.core import types
# from requirements import *

# @njit
# def compute_bash_loc(x, y, z, reactor_size):
#     return (x * reactor_size + y) * reactor_size + z

# @njit
# def find_loc_and_move_cells(pos_x, pos_y, pos_z, 
#                             last_off, 
#                             occupancy, 
#                             substrate, 
#                             offsets, 
#                             reactor_size, 
#                             feed_count, 
#                             consumed_substrate,
#                             released_electrons,
#                             consumed_substrate_locations):
#     """
#     1) For each agent in `pos`, try offsets[start:] to find a free, food‐stocked spot.
#     2) If found: vacate old idx, occupy new idx, consume substrate there, update pos & last_off.
#     """
#     N = len(pos_x)
#     M = len(offsets)
#     temp_electrons = 0

#     for i in range(N):
#         x, y, z = pos_x[i], pos_y[i], pos_z[i]
#         base = compute_bash_loc(x, y, z, reactor_size)
#         start = last_off[i] % M
#         for k in range(M):
#             dx, dy, dz = offsets[(start + k) % M]
#             nx, ny, nz = x + dx, y + dy, z + dz
#             if not (0 <= nx < reactor_size and 0 <= ny < reactor_size and 0 <= nz < reactor_size):
#                 continue
            
#             idx2 = compute_bash_loc(nx, ny, nz, reactor_size)
#             '''
#             occupancy[idx2] = 0 means free location, 1 means organism present
#             substrate[idx2] = 0 means food is absent, 1 means food is present
#             '''
#             # if occupancy.get(idx2, 0) == 0 and substrate.get(idx2, 0) == 1:
#             #     # Move + consume
#             #     occupancy.pop(base)
#             #     occupancy[idx2] = 1
#             #     substrate[idx2] = 0
#             #     pos[i, 0], pos[i, 1], pos[i, 2] = nx, ny, nz
#             #     last_off[i] = (start + k + 1) % M
#             #     break
#             # print('issue here????')

#             if idx2 not in consumed_substrate_locations:
#                 substrate[idx2] = initial_substrate_per_voxel
#             else:
#                 substrate[idx2] = 0
                

#             if occupancy.get(idx2, np.int8(0)) == 0 and substrate.get(idx2, np.int8(0)) > 0:
#                 # move
#                 if base in occupancy:
#                     occupancy.pop(base)
#                 occupancy[idx2] = 1
#                 pos_x[i], pos_y[i], pos_z[i] = nx, ny, nz
#                 last_off[i] = (start + k + 1) % M
#                 # eat & count
#                 substrate[idx2] -= 1
#                 consumed_substrate_locations[idx2] = 0
                
#                 consumed_substrate+=1
#                 temp_electrons += per_cell_electrons
#                 feed_count[i] += 1
#                 break
#             # print('end')
#     released_electrons = coulombic_efficiency * temp_electrons
#     return pos_x, pos_y, pos_z, occupancy, substrate, feed_count, consumed_substrate, released_electrons, consumed_substrate_locations

# @njit
# def get_neighbor_loc(x, y, z, offsets, reactor_size):

#     """
#     Get the locations of neighbors based on the given offsets.
#     """
#     M = offsets.shape[0]
#     out_idxs = np.empty(M, dtype=np.int64)
#     cnt = 0
#     for k in range(M):
#         dx, dy, dz = offsets[k]
#         nx, ny, nz = x + dx, y + dy, z + dz
#         if 0 <= nx < reactor_size and 0 <= ny < reactor_size and 0 <= nz < reactor_size:
#             idx2 = compute_bash_loc(nx, ny, nz, reactor_size)
#             out_idxs[cnt] = idx2
#             cnt += 1
#     return out_idxs[:cnt]  # slice to actual length

# @njit
# def births_and_deaths(organism_run_time, organism_growth_rate, offsets, feed_count, last_off, occupancy, substrate, organism_number, pos_x, pos_y, pos_z):
#     # print('1')
#     # try:
            
#     new_positions = []
#     dead = []
#     growth_rate = []
#     for i in range(organism_number):
#         organism_run_time[i] += 1
#         dt = np.log(2) / (organism_growth_rate[i] + 1e-9) * 3600/time_step
#         if feed_count[i] >= dt:
#             # reproduce: pick one of your offsets
#             dx,dy,dz = offsets[last_off[i] % len(offsets)]
#             x,y,z = pos_x[i], pos_y[i], pos_z[i]
#             nx, ny, nz = x+dx, y+dy, z+dz
#             idx2 = compute_bash_loc(nx, ny, nz, reactor_size)
#             if occupancy.get(idx2,np.int8(0))==0:
#                 new_positions.append((nx,ny,nz))
#                 feed_count[i] = 0  # reset parent
#                 growth_rate.append(organism_growth_rate[i])
#         elif organism_run_time[i] >= (np.log(2) * death_rate_percentage / organism_growth_rate[i]):
#             dead.append(i)
#             # print(' ----------------- ')
#         # except:
#         #     pass
#     # print('2 -------------------')
#     # Remove dead by swap‑pop
#     if dead:
#         for i in sorted(dead, reverse=True):
#             # print(i, organism_number)
#             x,y,z = pos_x[i], pos_y[i], pos_z[i]
#             idx = compute_bash_loc(x,y,z, reactor_size)
#             occupancy.pop(idx)
#             # print('here')
#             # swap i ↔ last-1
#             last = organism_number - 1
#             pos_x.pop(i)
#             pos_y.pop(i)
#             pos_z.pop(i)
#             # print('--------or here-------')
#             last_off.pop(i)
#             # print('1')
#             feed_count.pop(i)
#             # print('2')
#             organism_growth_rate.pop(i)
#             # print('3')
#             organism_run_time.pop(i)
#             organism_number -= 1
#             # print('___________ finally __________')

#     # print('3')
#     # Add daughters
#     for (nx,ny,nz) in new_positions:
#         i=0
#         idx = compute_bash_loc(nx, ny, nz, reactor_size)
#         if occupancy.get(idx,np.int8(0))==0:
#             # pos = list(pos)
#             # np.array(pos.append(np.array([nx,ny,nz])))
#             pos_x.append(nx)
#             pos_y.append(ny)
#             pos_z.append(nz)
#             # last_off = list(last_off)
#             # np.array(last_off.append(np.array(0)), dtype=np.int32)
#             last_off.append(0)
#             # feed_count = list(feed_count)
#             # np.array(feed_count.append(0), dtype=np.int8)
#             feed_count.append(0)
#             # organism_growth_rate = list(organism_growth_rate)
#             # np.array(organism_growth_rate.append(growth_rate[i] * (1 + np.random.uniform(-0.05, 0.05))), dtype=np.float16)
#             organism_growth_rate.append(growth_rate[i] * (1 + np.random.uniform(-0.05, 0.05)))
#             # organism_run_time = list(organism_run_time)
#             # np.array(organism_run_time.append(0), dtype=np.int8)
#             organism_run_time.append(0)

#             occupancy[idx]      = 1
#             substrate[idx]      = initial_substrate_per_voxel
#             organism_number += 1
#         i+=1
#     # print('4')
#     # except:
#     #     pass
#     return organism_run_time, organism_growth_rate, offsets, feed_count, last_off, occupancy, substrate, pos_x, pos_y, pos_z, organism_number



import numpy as np
import pandas as pd
from numba import njit, int32, int64
from numba.typed import Dict, List
from numba.core import types
from requirements import *

@njit
def compute_bash_loc(x, y, z, reactor_size):
    return (x * reactor_size + y) * reactor_size + z

@njit
def find_loc_and_move_cells(pos_x, pos_y, pos_z, 
                            last_off, 
                            occupancy, 
                            substrate, 
                            offsets, 
                            reactor_size, 
                            feed_count, 
                            consumed_substrate,
                            released_electrons,
                            consumed_substrate_locations):
    """
    1) For each agent in `pos`, try offsets[start:] to find a free, food‐stocked spot.
    2) If found: vacate old idx, occupy new idx, consume substrate there, update pos & last_off.
    """
    N = len(pos_x)
    M = len(offsets)
    temp_electrons = 0

    for i in range(N):
        x, y, z = pos_x[i], pos_y[i], pos_z[i]
        base = compute_bash_loc(x, y, z, reactor_size)
        start = last_off[i] % M
        for k in range(M):
            dx, dy, dz = offsets[(start + k) % M]
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < reactor_size and 0 <= ny < reactor_size and 0 <= nz < reactor_size):
                continue
            
            idx2 = compute_bash_loc(nx, ny, nz, reactor_size)
            '''
            occupancy[idx2] = 0 means free location, 1 means organism present
            substrate[idx2] = 0 means food is absent, 1 means food is present
            '''
            # if occupancy.get(idx2, 0) == 0 and substrate.get(idx2, 0) == 1:
            #     # Move + consume
            #     occupancy.pop(base)
            #     occupancy[idx2] = 1
            #     substrate[idx2] = 0
            #     pos[i, 0], pos[i, 1], pos[i, 2] = nx, ny, nz
            #     last_off[i] = (start + k + 1) % M
            #     break
            # print('issue here????')

            if idx2 not in consumed_substrate_locations:
                substrate[idx2] = initial_substrate_per_voxel
            else:
                substrate[idx2] = 0
                

            if occupancy.get(idx2, np.int8(0)) == 0 and substrate.get(idx2, np.int8(0)) > 0:
                # move
                if base in occupancy:
                    occupancy.pop(base)
                occupancy[idx2] = 1
                pos_x[i], pos_y[i], pos_z[i] = nx, ny, nz
                last_off[i] = (start + k + 1) % M
                # eat & count
                substrate[idx2] -= 1
                consumed_substrate_locations[idx2] = 0
                
                consumed_substrate+=1
                temp_electrons += per_cell_electrons
                feed_count[i] += 1
                break
            # print('end')
    released_electrons = coulombic_efficiency * temp_electrons
    return pos_x, pos_y, pos_z, occupancy, substrate, feed_count, consumed_substrate, released_electrons, consumed_substrate_locations

@njit
def get_neighbor_loc(x, y, z, offsets, reactor_size):

    """
    Get the locations of neighbors based on the given offsets.
    """
    M = offsets.shape[0]
    out_idxs = np.empty(M, dtype=np.int64)
    cnt = 0
    for k in range(M):
        dx, dy, dz = offsets[k]
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < reactor_size and 0 <= ny < reactor_size and 0 <= nz < reactor_size:
            idx2 = compute_bash_loc(nx, ny, nz, reactor_size)
            out_idxs[cnt] = idx2
            cnt += 1
    return out_idxs[:cnt]  # slice to actual length


@njit
def births_and_deaths(organism_run_time, organism_growth_rate, offsets, feed_count, last_off, occupancy, substrate, organism_number, pos_x, pos_y, pos_z):
    # print('1')
    # try:
            
    new_positions = []
    dead = []
    growth_rate = []
    for i in range(organism_number):
        # try:

        organism_run_time[i] += 1
        dt = np.log(2) / (organism_growth_rate[i] + 1e-9) * 3600/time_step
        if feed_count[i] >= dt:
            # reproduce: pick one of your offsets
            dx,dy,dz = offsets[last_off[i] % len(offsets)]
            x,y,z = pos_x[i], pos_y[i], pos_z[i]
            nx, ny, nz = x+dx, y+dy, z+dz
            idx2 = compute_bash_loc(nx, ny, nz, reactor_size)
            if occupancy.get(idx2,np.int8(0))==0:
                new_positions.append((nx,ny,nz))
                feed_count[i] = 0  # reset parent
                growth_rate.append(organism_growth_rate[i])
        elif organism_run_time[i] >= (np.log(2) * death_rate_percentage / organism_growth_rate[i]):
            dead.append(i)
        # except:
        #     pass
    # print('2 -------------------')
    # Remove dead by swap‑pop
    if dead:
        for i in sorted(dead, reverse=True):
            # print(i, organism_number)
            x,y,z = pos_x[i], pos_y[i], pos_z[i]
            idx = compute_bash_loc(x,y,z, reactor_size)
            occupancy.pop(idx)
            # print('here')
            # swap i ↔ last-1
            last = organism_number - 1
            pos_x.pop(i)
            pos_y.pop(i)
            pos_z.pop(i)
            # print('--------or here-------')
            last_off.pop(i)
            # print('1')
            feed_count.pop(i)
            # print('2')
            organism_growth_rate.pop(i)
            # print('3')
            organism_run_time.pop(i)
            organism_number -= 1
            # print('___________ finally __________')

    # print('3')
    # Add daughters
    for (nx,ny,nz) in new_positions:
        i=0
        idx = compute_bash_loc(nx, ny, nz, reactor_size)
        if occupancy.get(idx,np.int8(0))==0:
            # pos = list(pos)
            # np.array(pos.append(np.array([nx,ny,nz])))
            pos_x.append(nx)
            pos_y.append(ny)
            pos_z.append(nz)
            # last_off = list(last_off)
            # np.array(last_off.append(np.array(0)), dtype=np.int32)
            last_off.append(0)
            # feed_count = list(feed_count)
            # np.array(feed_count.append(0), dtype=np.int8)
            feed_count.append(0)
            # organism_growth_rate = list(organism_growth_rate)
            # np.array(organism_growth_rate.append(growth_rate[i] * (1 + np.random.uniform(-0.05, 0.05))), dtype=np.float16)
            organism_growth_rate.append(growth_rate[i] * (1 + np.random.uniform(-0.05, 0.05)))
            # organism_run_time = list(organism_run_time)
            # np.array(organism_run_time.append(0), dtype=np.int8)
            organism_run_time.append(0)

            occupancy[idx]      = 1
            substrate[idx]      = initial_substrate_per_voxel
            organism_number += 1
        i+=1
    # print('4')
    # except:
    #     pass
    return organism_run_time, organism_growth_rate, offsets, feed_count, last_off, occupancy, substrate, pos_x, pos_y, pos_z, organism_number