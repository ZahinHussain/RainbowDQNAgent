import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

def make_env(env_id: str):
    is_atari = env_id.startswith("ALE/")

    if is_atari:
        base_env = gym.make(env_id, frameskip=1)
        env = AtariPreprocessing(
            base_env,
            grayscale_obs=True,
            scale_obs=True,
            frame_skip=4,
            screen_size=84
        )
        env = FrameStackObservation(env, stack_size=4)
    else:
        env = gym.make(env_id)

    return env
