import os
import warnings
import retrowrapper

from collections import OrderedDict
from typing import Sequence
from copy import deepcopy

from gym import spaces
from stable_baselines.common.vec_env.base_vec_env import VecEnv
from stable_baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from stable_baselines.bench import Monitor
from collections import namedtuple
import numpy as np
from enum import Enum, unique

# Section from Chrispresso @ GitHub begins (Some edits were made)
@unique
class EnemyType(Enum):
    Green_Koopa1 = 0x00
    Red_Koopa1 = 0x01
    Buzzy_Beetle = 0x02
    Red_Koopa2 = 0x03
    Green_Koopa2 = 0x04
    Hammer_Brother = 0x05
    Goomba = 0x06
    Blooper = 0x07
    Bullet_Bill = 0x08
    Green_Koopa_Paratroopa = 0x09
    Grey_Cheep_Cheep = 0x0A
    Red_Cheep_Cheep = 0x0B
    Pobodoo = 0x0C
    Piranha_Plant = 0x0D
    Green_Paratroopa_Jump = 0x0E
    Bowser_Flame1 = 0x10
    Lakitu = 0x11
    Spiny_Egg = 0x12
    Fly_Cheep_Cheep = 0x14
    Bowser_Flame2 = 0x15

    Generic_Enemy = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)


@unique
class StaticTileType(Enum):
    Empty = 0x00
    Fake = 0x01
    Ground = 0x54
    Top_Pipe1 = 0x12
    Top_Pipe2 = 0x13
    Bottom_Pipe1 = 0x14
    Bottom_Pipe2 = 0x15
    Flagpole_Top = 0x24
    Flagpole = 0x25
    Coin_Block1 = 0xC0
    Coin_Block2 = 0xC1
    Coin = 0xC2
    Breakable_Block = 0x51

    Generic_Static_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)


@unique
class DynamicTileType(Enum):
    Mario = 0xAA

    Static_Lift1 = 0x24
    Static_Lift2 = 0x25
    Vertical_Lift1 = 0x26
    Vertical_Lift2 = 0x27
    Horizontal_Lift = 0x28
    Falling_Static_Lift = 0x29
    Horizontal_Moving_Lift = 0x2A
    Lift1 = 0x2B
    Lift2 = 0x2C
    Vine = 0x2F
    Flagpole = 0x30
    Start_Flag = 0x31
    Jump_Spring = 0x32
    Warpzone = 0x34
    Spring1 = 0x67
    Spring2 = 0x68

    Generic_Dynamic_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)


class ColorMap(Enum):
    Empty = (255, 255, 255)  # White
    Ground = (128, 43, 0)  # Brown
    Fake = (128, 43, 0)
    Mario = (0, 0, 255)
    Goomba = (255, 0, 20)
    Top_Pipe1 = (0, 15, 21)  # Dark Green
    Top_Pipe2 = (0, 15, 21)  # Dark Green
    Bottom_Pipe1 = (5, 179, 34)  # Light Green
    Bottom_Pipe2 = (5, 179, 34)  # Light Green
    Coin_Block1 = (219, 202, 18)  # Gold
    Coin_Block2 = (219, 202, 18)  # Gold
    Jump_Spring = (219, 202, 18)  # Gold
    Breakable_Block = (79, 70, 25)  # Brownish

    Generic_Enemy = (255, 0, 20)  # Red
    Generic_Static_Tile = (128, 43, 0)
    Generic_Dynamic_Tile = (79, 70, 25)


Shape = namedtuple('Shape', ['width', 'height'])
Point = namedtuple('Point', ['x', 'y'])


class Tile(object):
    __slots__ = ['type']

    def __init__(self, type: Enum):
        self.type = type


class Enemy(object):
    def __init__(self, enemy_id: int, location: Point, tile_location: Point):
        self.type = EnemyType(enemy_id)
        self.location = location
        self.tile_location = tile_location


class SMB(object):
    # SMB can only load 5 enemies to the screen at a time.
    # Because of that we only need to check 5 enemy locations
    MAX_NUM_ENEMIES = 5
    PAGE_SIZE = 256
    NUM_BLOCKS = 8
    RESOLUTION = Shape(256, 240)
    NUM_TILES = 416  # 0x69f - 0x500 + 1
    NUM_SCREEN_PAGES = 2
    TOTAL_RAM = NUM_BLOCKS * PAGE_SIZE

    sprite = Shape(width=16, height=16)
    resolution = Shape(256, 240)
    status_bar = Shape(width=resolution.width, height=2 * sprite.height)

    xbins = list(range(16, resolution.width, 16))
    ybins = list(range(16, resolution.height, 16))

    @unique
    class RAMLocations(Enum):
        # Since the max number of enemies on the screen is 5, the addresses for enemies are
        # the starting address and span a total of 5 bytes. This means Enemy_Drawn + 0 is the
        # whether or not enemy 0 is drawn, Enemy_Drawn + 1 is enemy 1, etc. etc.
        Enemy_Drawn = 0x0F
        Enemy_Type = 0x16
        Enemy_X_Position_In_Level = 0x6E
        Enemy_X_Position_On_Screen = 0x87
        Enemy_Y_Position_On_Screen = 0xCF

        Player_X_Postion_In_Level = 0x06D
        Player_X_Position_On_Screen = 0x086

        Player_X_Position_Screen_Offset = 0x3AD
        Player_Y_Position_Screen_Offset = 0x3B8
        Enemy_X_Position_Screen_Offset = 0x3AE

        Player_Y_Pos_On_Screen = 0xCE
        Player_Vertical_Screen_Position = 0xB5

    @classmethod
    def get_enemy_locations(cls, ram: np.ndarray):
        # We only care about enemies that are drawn. Others may?? exist
        # in memory, but if they aren't on the screen, they can't hurt us.
        # enemies = [None for _ in range(cls.MAX_NUM_ENEMIES)]
        enemies = []

        for enemy_num in range(cls.MAX_NUM_ENEMIES):
            enemy = ram[cls.RAMLocations.Enemy_Drawn.value + enemy_num]
            # Is there an enemy? 1/0
            if enemy:
                # Get the enemy X location.
                x_pos_level = ram[cls.RAMLocations.Enemy_X_Position_In_Level.value + enemy_num]
                x_pos_screen = ram[cls.RAMLocations.Enemy_X_Position_On_Screen.value + enemy_num]
                enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen  # - ram[0x71c]
                # print(ram[0x71c])
                # enemy_loc_x = ram[cls.RAMLocations.Enemy_X_Position_Screen_Offset.value + enemy_num]
                # Get the enemy Y location.
                enemy_loc_y = ram[cls.RAMLocations.Enemy_Y_Position_On_Screen.value + enemy_num]
                # Set location
                location = Point(enemy_loc_x, enemy_loc_y)
                ybin = np.digitize(enemy_loc_y, cls.ybins)
                xbin = np.digitize(enemy_loc_x, cls.xbins)
                tile_location = Point(xbin, ybin)

                # Grab the id
                enemy_id = ram[cls.RAMLocations.Enemy_Type.value + enemy_num]
                # Create enemy-
                e = Enemy(enemy_id, location, tile_location)

                enemies.append(e)

        return enemies

    @classmethod
    def get_mario_location_in_level(cls, ram: np.ndarray) -> Point:
        mario_x = ram[cls.RAMLocations.Player_X_Postion_In_Level.value] * 256 + ram[
            cls.RAMLocations.Player_X_Position_On_Screen.value]
        mario_y = ram[cls.RAMLocations.Player_Y_Position_Screen_Offset.value]
        return Point(mario_x, mario_y)

    @classmethod
    def get_mario_score(cls, ram: np.ndarray) -> int:
        multipllier = 10
        score = 0
        for loc in range(0x07DC, 0x07D7 - 1, -1):
            score += ram[loc] * multipllier
            multipllier *= 10

        return score

    @classmethod
    def get_mario_location_on_screen(cls, ram: np.ndarray):
        mario_x = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value]
        mario_y = ram[cls.RAMLocations.Player_Y_Pos_On_Screen.value] * int(ram[
                                                                               cls.RAMLocations.Player_Vertical_Screen_Position.value]) + int(
            cls.sprite.height)
        return Point(mario_x, mario_y)

    @classmethod
    def get_tile_type(cls, ram: np.ndarray, delta_x: int, delta_y: int, mario: Point):
        x = mario.x + delta_x
        y = mario.y + delta_y + cls.sprite.height

        # Tile locations have two pages. Determine which page we are in
        page = (x // 256) % 2
        # Figure out where in the page we are
        sub_page_x = (x % 256) // 16
        sub_page_y = (y - 32) // 16  # The PPU is not part of the world, coins, etc (status bar at top)
        if sub_page_y not in range(13):  # or sub_page_x not in range(16):
            return StaticTileType.Empty.value

        addr = 0x500 + page * 208 + sub_page_y * 16 + sub_page_x
        return ram[addr]

    @classmethod
    def get_tile_loc(cls, x, y):
        row = np.digitize(y, cls.ybins) - 2
        col = np.digitize(x, cls.xbins)
        return (row, col)

    # Empty = 0, Ground = 1, Generic Enemy = -1, Jump Spring = -2, Mario = 170
    # Ex. [1,0,0,0,0] = Empty
    @classmethod
    def get_tiles(cls, ram: np.ndarray):
        tiles = {}
        row = 0
        col = 0

        mario_level = cls.get_mario_location_in_level(ram)
        mario_screen = cls.get_mario_location_on_screen(ram)

        x_start = mario_level.x - mario_screen.x

        enemies = cls.get_enemy_locations(ram)
        y_start = 0
        mx, my = cls.get_mario_location_in_level(ram)
        my += 16
        # Set mx to be within the screen offset
        mx = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value]

        for y_pos in range(y_start, 240, 16):
            for x_pos in range(x_start, x_start + 256, 16):
                loc = (row, col)
                tile = cls.get_tile(x_pos, y_pos, ram)
                x, y = x_pos, y_pos
                page = (x // 256) % 2
                sub_x = (x % 256) // 8
                sub_y = (y - 32) // 8
                addr = 0x500 + page * 208 + sub_y * 16 + sub_x

                # PPU is there, so no tile is there
                if row < 2:
                    tiles[loc] = [1, 0, 0, 0]  # StaticTileType.Empty

                else:

                    if not StaticTileType(tile) == StaticTileType.Empty \
                            and not StaticTileType(tile) == StaticTileType.Fake:
                        tiles[loc] = [0, 1, 0, 0]  # Ground
                    else:
                        tiles[loc] = [1, 0, 0, 0]  # Air/Empty

                    for enemy in enemies:
                        ex = enemy.location.x
                        ey = enemy.location.y + 8
                        # Since we can only discriminate within 8 pixels, if it falls within this bound, count it as there
                        if abs(x_pos - ex) <= 8 and abs(y_pos - ey) <= 8:
                            # Jump Spring is seen as an enemy, so this has to be done to allow the AI to see a jump spring
                            if enemy.type == 0x32:
                                tiles[loc] = [0, 0, 1, 0]  # DynamicTileType.Jump_Spring
                            else:
                                tiles[loc] = [0, 0, 1, 0]  # Generic Enemy
                # Next col
                col += 1
            # Move to next row
            col = 0
            row += 1

        # Place marker for mario
        mario_row, mario_col = cls.get_mario_row_col(ram)
        loc = (mario_row, mario_col)
        tiles[loc] = [0, 0, 0, 1]  # DynamicTileType.Mario

        return tiles

    @classmethod
    def get_mario_row_col(cls, ram):
        x, y = cls.get_mario_location_on_screen(ram)
        # Adjust 16 for PPU
        y = ram[cls.RAMLocations.Player_Y_Position_Screen_Offset.value] + 16
        x += 12
        col = x // 16
        row = (y - 0) // 16
        return (row, col)

    @classmethod
    def get_tile(cls, x, y, ram, group_non_zero_tiles=True):
        page = (x // 256) % 2
        sub_x = (x % 256) // 16
        sub_y = (y - 32) // 16

        if sub_y not in range(13):
            return StaticTileType.Empty.value

        addr = 0x500 + page * 208 + sub_y * 16 + sub_x
        if group_non_zero_tiles:
            if ram[addr] != 0 and ram[addr] != 0xC2:
                return StaticTileType.Ground.value

        return ram[addr]
# Section from Chrispresso @ GitHub ends


def make_vec_env_mario(env_id, n_envs=1, seed=None, start_index=0,
                       monitor_dir=None, wrapper_class=None,
                       env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None):
    """
    Create a wrapped, monitored `VecEnv`.
    By default it uses a `DummyVecEnv` which is usually faster
    than a `SubprocVecEnv`.

    :param env_id: (str or Type[gym.Env]) the environment ID or the environment class
    :param n_envs: (int) the number of environments you wish to have in parallel
    :param seed: (int) the initial seed for the random number generator
    :param start_index: (int) start rank index
    :param monitor_dir: (str) Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: (dict) Optional keyword argument to pass to the env constructor
    :param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.
    :param vec_env_kwargs: (dict) Keyword arguments to pass to the `VecEnv` class constructor.
    :return: (VecEnv) The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = retrowrapper.RetroWrapper(env_id)
                env.reset()
                if len(env_kwargs) > 0:
                    warnings.warn("No environment class was passed (only an env ID) so `env_kwargs` will be ignored")
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnvMario

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


class DummyVecEnvMario(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``, as the overhead of
    multiprocess or multithread outweighs the environment computation time. This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), spaces.Box(0, 1, shape=(14, 14, 4), dtype=int), env.action_space)
        obs_space = spaces.Box(0, 1, shape=(14, 14, 4), dtype=int)
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([
            (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
            for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata
        self.frames_since_max_x_change = np.zeros((self.num_envs,), dtype=np.float32)
        self.max_x_change = np.zeros((self.num_envs,), dtype=np.float32)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = \
                self.envs[env_idx].step(self.actions[env_idx])

            # Added section
            ram = self.envs[env_idx].get_ram()
            obs = get_input_array(ram)
            mario_x = SMB.get_mario_location_in_level(ram)[0]
            if mario_x > self.max_x_change[env_idx]:
                self.max_x_change[env_idx] = mario_x
                self.frames_since_max_x_change[env_idx] = 0
            else:
                self.frames_since_max_x_change[env_idx] += 1
            if self.frames_since_max_x_change[env_idx] > 125:
                self.buf_dones[env_idx] = True
            player_state = ram[0x000E]  # Check if dead
            player_float_state = ram[0x001D]  # Check if finished level
            if player_state == 0x06 or player_state == 0x0B:
                self.buf_dones[env_idx] = True
            if player_float_state == 0x03:  # Is Mario on flag?
                self.buf_dones[env_idx] = True

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.max_x_change[env_idx] = 0
                self.frames_since_max_x_change[env_idx] = 0
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs = self.envs[env_idx].reset()
                ram = self.envs[env_idx].get_ram()
                obs = get_input_array(ram)
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                deepcopy(self.buf_infos))

    def seed(self, seed=None):
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            ram = self.envs[env_idx].get_ram()
            obs = get_input_array(ram)
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self):
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode: str = 'human'):
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via `BaseVecEnv.render()`.
        Otherwise (if `self.num_envs == 1`), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as `mode` will have values that are valid
        only when `num_envs == 1`.

        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]


def get_input_array(ram):
    tiles = SMB.get_tiles(ram)  # Get the tiles from the ram
    input_array = []
    temp_array = []
    for i in range(1, 15):  # From 0 to 15
        for j in range(1, 15):  # From 0 to 15
            temp_array.append(tiles[i, j])
        input_array.append(temp_array)
        temp_array = []
    input_array = np.array(input_array)
    return input_array
