import gymnasium as gym
import numpy as np
import pydash as ps

from tac.utils import general

NUM_EVAL = 100
FILEPATH = "tac/spec/_random_baseline.json"

INCLUDE_ENVS = [
    'vizdoom-v0',
]

EXCLUDE_ENVS = [
    "GymV21Environment-v0",    # New gymnasium addition
    "GymV26Environment-v0",    # New gymnasium addition
    "CarRacing-v0",  # window bug
    "Reacher-v2",  # exclude mujoco
    "Pusher-v2",
    "Thrower-v2",
    "Striker-v2",
    "InvertedPendulum-v2",
    "InvertedDoublePendulum-v2",
    "HalfCheetah-v3",
    "Hopper-v3",
    "Swimmer-v3",
    "Walker2d-v3",
    "Ant-v3",
    "Humanoid-v3",
    "HumanoidStandup-v2",
    "FetchSlide-v1",
    "FetchPickAndPlace-v1",
    "FetchReach-v1",
    "FetchPush-v1",
    "HandReach-v0",
    "HandManipulateBlockRotateZ-v0",
    "HandManipulateBlockRotateParallel-v0",
    "HandManipulateBlockRotateXYZ-v0",
    "HandManipulateBlockFull-v0",
    "HandManipulateBlock-v0",
    "HandManipulateBlockTouchSensors-v0",
    "HandManipulateEggRotate-v0",
    "HandManipulateEggFull-v0",
    "HandManipulateEgg-v0",
    "HandManipulateEggTouchSensors-v0",
    "HandManipulatePenRotate-v0",
    "HandManipulatePenFull-v0",
    "HandManipulatePen-v0",
    "HandManipulatePenTouchSensors-v0",
    "FetchSlideDense-v1",
    "FetchPickAndPlaceDense-v1",
    "FetchReachDense-v1",
    "FetchPushDense-v1",
    "HandReachDense-v0",
    "HandManipulateBlockRotateZDense-v0",
    "HandManipulateBlockRotateParallelDense-v0",
    "HandManipulateBlockRotateXYZDense-v0",
    "HandManipulateBlockFullDense-v0",
    "HandManipulateBlockDense-v0",
    "HandManipulateBlockTouchSensorsDense-v0",
    "HandManipulateEggRotateDense-v0",
    "HandManipulateEggFullDense-v0",
    "HandManipulateEggDense-v0",
    "HandManipulateEggTouchSensorsDense-v0",
    "HandManipulatePenRotateDense-v0",
    "HandManipulatePenFullDense-v0",
    "HandManipulatePenDense-v0",
    "HandManipulatePenTouchSensorsDense-v0",
]


def enum_envs():
    """Enumerate all environments in the gym registry"""
    envs = []
    for env_name in gym.envs.registration.registry.keys():
        envs.append(env_name)
    # Filter out older versions (combination of reverse and uniq_by ensures that the newest version is kept)
    def get_name(s): return s.split("-")[0]
    envs = ps.reverse(ps.uniq_by(envs, get_name))
    # Filter out excluded envs
    envs = ps.difference_by(envs, EXCLUDE_ENVS, get_name)
    # Filter in included envs
    envs += INCLUDE_ENVS
    return envs


def gen_random_return(env_name, seed):
    """Generate a single-episode random policy return for an environment"""
    env = gym.make(env_name)
    _, _ = env.reset(seed=seed, options={})
    done = False
    total_reward = 0
    while not done:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        total_reward += reward
        done = terminated or truncated
    return total_reward


def gen_random_baseline(env_name, num_eval=NUM_EVAL):
    """Generate a random policy baseline for an environment by averaging over num_eval episodes"""
    random_returns = []
    for seed in range(num_eval):
        random_returns.append(gen_random_return(env_name, seed))
    mean_return = np.mean(random_returns)
    std_return = np.std(random_returns)
    return {"mean_return": mean_return, "std_return": std_return}


def get_random_baseline(env_name):
    """Get a single random baseline for an environment. If it doesn't exist in file, generate it live and update file"""
    random_baseline = general.read(data_path=FILEPATH)
    if env_name in random_baseline:
        baseline = random_baseline[env_name]
    else:
        try:
            baseline = gen_random_baseline(env_name)
        except Exception as e:
            print(f"Error generating random baseline for {env_name}: {e}")
            baseline = None
        random_baseline[env_name] = baseline
        general.write(data=random_baseline, data_path=FILEPATH)
    return baseline


def main():
    """Main method to generate all random baselines and write to file"""
    envs = enum_envs()
    for idx, env_name in enumerate(envs):
        print(f"Generating random baseline for {env_name} ({idx+1}/{len(envs)})")
        get_random_baseline(env_name)


if __name__ == "__main__":
    main()
