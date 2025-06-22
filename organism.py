import mesa
from PreCode import *

class organism(mesa.agent):
    def __init__(self, model, pos):
        super().__init__(model)
        self.pos = pos

    def feed_and_move(self, pos):
        '''
        organism moves to new place, finds food consumes it and initiate reproduction
        '''
        # Movement and substrate consumption
        find_loc_and_move_cells(
            self.pos,
            self.last_off,
            self.model.occupancy,
            self.model.substrate,
            self.offsets,
            self.reactor_size
        )
        pass

    def general_reproduce():
        pass

    def planktonic_reproduce():
        pass

    def biofilm_reproduce():
        pass

    def step():
        pass
