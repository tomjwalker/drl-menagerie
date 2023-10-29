from abc import ABC, abstractmethod
import time


class Clock:

    def __init__(self, max_frame=int(1e7), clock_speed=1):

        self.max_frame = max_frame
        self.clock_speed = int(clock_speed)

        # These next attributes are set in the reset method
        self.opt_step = None
        self.batch_size = None
        self.wall_t = None
        self.start_wall_t = None
        self.epi = None
        self.frame = None
        self.t = None
        self.reset()

    def reset(self):
        self.t = 0
        self.frame = 0    # i.e. total_t
        self.epi = 0
        # Register the time at which the clock was reset (in seconds)
        self.start_wall_t = time.time()
        self.wall_t = 0
        self.batch_size = 1    # Multiplier to accurately count opt steps
        self.opt_step = 0    # Count the number of optimiser updates

    def load(self, train_df):
        raise NotImplementedError

    def get(self, unit="frame"):
        raise getattr(self, unit)

    def get_elapsed_wall_time(self):
        """
        Calling time.time() again gets the current time, and subtracting the start time gives the elapsed time
        (in seconds)
        """
        return int(time.time() - self.start_wall_t)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def tick(self, unit="t"):
        if unit == "t":    # timestep
            self.t += self.clock_speed    # Reset to 0 at the start of each episode
            self.frame += self.clock_speed    # Not reset to 0 at the start of each episode, so tracks total timesteps
            self.wall_t = self.get_elapsed_wall_time()
        elif unit == "epi":    # Episode; reset timestep counter t
            self.epi += 1
            self.t = 0
        elif unit == "opt_step":    # Optimisation step
            self.opt_step += self.batch_size
        else:
            raise KeyError(f"Unit {unit} not recognised")


class BaseEnv(ABC):
    raise NotImplementedError
