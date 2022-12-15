import ray
import os
import sys
from ray.rllib.algorithms import ppo
from ray.tune.logger import pretty_print

from cluster import Cluster
from model import TorchActionMaskModel

PATH = "data/"


config = (
    ppo.PPOConfig().environment(
        Cluster,
        env_config={"workload": PATH + "lublin_256.swf"},
        disable_env_checking=True,
    ).training(
        # the ActionMaskModel retrieves the invalid actions and avoids them
        model={
            "custom_model": TorchActionMaskModel,
            "custom_model_config": {
                "fcnet_hiddens": [1024, 1024],
            }
        },
        lr=1e-5
    ).rollouts(
        num_rollout_workers=4,
        num_envs_per_worker=1
    ).framework("torch")
)

algo = config.build()

epochs = 2000
for _ in range(epochs):
    result = algo.train()
    print(pretty_print(result))

# evaluation
env = Cluster(config.env_config)
obs = env.reset()
done = False
while not done:
    action = algo.compute_single_action(obs)
    obs, reward, done, _ = env.step(action)

ray.shutdown()
print("test reward: ", reward)

# random evaluation
# test loop
c = Cluster({"workload": PATH + "lublin_256.swf"})
init = c.reset()
obs = torch.tensor(init["obs"])
mask = torch.tensor(init["mask"])
done = False
reward = 0
while not done:
    logits = torch.randn(QUEUE_SIZE + 1)  # should be from forward pass
    inf_mask = torch.clamp(torch.log(mask), min=FLOAT_MIN)
    masked_logits = logits + inf_mask
    action = torch.argmax(masked_logits)
    obs_dict, reward, done, _ = c.step(action)

    obs = torch.tensor(obs_dict["obs"])
    mask = torch.tensor(obs_dict["mask"])

print("random reward: ", reward)

recorded = 0
for j in c.workload:
    recorded += j.recorded_wait_time

print("recorded wait: ", recorded / len(c.workload))
