from .RaceAviary import RaceAviary, DeployType
from gymnasium.envs.registration import register
from .utils import DecreaseOmegaCoef

register(
    id='race-aviary-v0',
    entry_point=RaceAviary
)