from dataclasses import dataclass, field
import numpy as np
import uuid
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
import copy

def fx(state, dt):
    x, y, z, vx, vy, vz = state
    return [x + vx*dt, y + vy*dt, z + vz*dt, vx, vy, vz]

def hx(state):
    return state[:3]


@dataclass
class Target:
    _positions: np.ndarray = field(default_factory=lambda: np.empty((0,3)))
    _timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    _uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    _class: str = field(default_factory=lambda: str('Unknown'))
    _ukf: UKF = field(init=False)
    _future_prediction: np.ndarray = field(default_factory=lambda: np.empty((0,3))) # why haven't I used this? I don't know
    _n_missed_frames: int = field(default=0)
    _max_missed_frames: int = field(default=3)

    def __post_init__(self):
        if len(self._timestamps) != len(self._positions):
            raise ValueError("When supplying initial positions and timestamps, they must have equal lengths")
        if len(self._timestamps) == 0 and len(self._positions) == 0:
            self._pos_initialized = False
            self._ukf_initialized = False
        else:
            self._pos_initialized = True
        self.invalid_target = False


    def init_ukf(self, dt):
        """
        After the first append, dt is 0 because it's the only position, p1, and nothing is done.
        After the second append, p2, init_ukf is called and the initial position to be used
        will be the p1, and its velocity from p1 to p2. 
        So the UKF will have an x_0 = p1 + v. Then do UKF.predict(), then UKF.update(p2)
        """
        p1 = self._positions[0]
        p2 = self._positions[1]
        x0, y0, z0 = p1
        x1, y1, z1 = p2 # Redundant but it's more clear - non issue
        velocity = np.array([(x1-x0)/dt, (y1-y0)/dt, (z1-z0)/dt])
        x = np.concatenate((p1, velocity), axis=None)

        self.points = MerweScaledSigmaPoints(n=6, alpha=.1, beta=2., kappa=0)
        self._ukf = UKF(dim_x=6, dim_z=3, fx=fx, hx=hx, points=self.points, dt=dt)
        #self._ukf.x = np.array([0.,0.,0.,0.,0.,0.])
        self._ukf.x = x
        self._ukf.P *= .2
        self._ukf.R *= np.diag([.1**2, .1**2, .1**2])
        self._ukf.Q = np.eye(6) * .1

        self.add_measurement(p2, dt)
        self._ukf_initialized = True

    # This should ONLY be called by append, add_pos, OR if for some reason
    # you're manually changing _positions and _timestamps and need to call this after
    def add_measurement(self, measurement, dt):
        self._ukf.dt = dt
        self._ukf.predict()
        self._ukf.update(measurement)

    def get_prediction(self):
        state_copy = copy.deepcopy(self._ukf)
        state_copy.predict()
        return (state_copy.x, self._ukf.x)

    def get_pos(self):
        return self._positions

    def get_times(self):
        return self._timestamps

    def get_uid(self):
        return self._uid

    def get_class(self):
        return self._class

    def add_pos(self, pos, dt):
        self.append(pos, dt)

    def no_match(self):
        self._n_missed_frames += 1
        if self._n_missed_frames >= self._max_missed_frames:
            self.invalid_target = True


    def append(self, pos, dt):
        if isinstance(pos, (list, tuple, np.ndarray)) :
            if not isinstance(dt, (int, float)):
                raise ValueError("dt must be a numerical value (int or float)")
            if len(pos) == 3:
                parr = np.array([pos])
                self._positions = np.vstack([self._positions, parr])

                if len(self._timestamps) == 0:
                    self._timestamps = np.append(self._timestamps, dt) # dt in this case SHOULD be 0
                else:
                    self._timestamps = np.append(self._timestamps, self._timestamps[-1] + dt)

                if self._pos_initialized == True:
                    """
                    x1,y1,z1 = pos
                    x0,y0,z0 = self._positions[-1]
                    velocity = [x1-x0, y1-y0, z1-z0]
                    measurement = np.append(pos, velocity)
                    self.add_measurement(measurement, dt)
                    """
                    if self._ukf_initialized == False:
                        self.init_ukf(dt)
                    else:
                        self.add_measurement(pos,dt)
                else:
                    #initial_arr = np.append(np.array(pos), [0,0,0]) # current position (x, y, z) + initial velocity 0 (0, 0, 0)
                    #self._ukf.x = initial_arr 
                    self._pos_initialized = True

            else:
                raise ValueError("Position must have exactly 3 elements (x, y, z)")
        else:
            raise TypeError("Position must be of type list, tuple, or numpy array")

    def pop(self, index=0):
        if not isinstance(index, (int)):
            raise ValueError("Index must be of type int")
        if index < 0 or index >= len(self._positions):
            raise IndexError("Index out of range")
        pos_pop = self._positions(index)
        time_pop = self._timestamps(index)
        self._positions = np.delete(self._positions, index, axis=0)
        self._timestamps = np.delete(self._timestamps, index)

    def __eq__(self, other):
        return isinstance(other, Target) and self._uid == other._uid

    def __hash__(self):
        return hash(self._uid)
            
    def __len__(self) -> int:
        return len(self._positions)
    
    def __contains__(self, item:np.ndarray) -> bool:
        return item in self._positions

    def __getitem__(self, index):
        if not isinstance(index, (int)):
            raise ValueError("Index must be of type int")
        if index < 0 or index >= len(self._positions):
            raise IndexError("Index out of range")
        return self._positions[index]

    ## TODO: Update everything below this line

    def __delitem__(self, item):
        pass


