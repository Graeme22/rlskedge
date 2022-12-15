from gym.Spaces import Box, Dict, Discrete
import numpy as np
from ray.rllib.env.env_context import EnvContext

from job import Workload

QUEUE_SIZE = 128
ZONES = 32


class Cluster(gym.Env):
    def __init__(self, config: EnvContext):
        # the action taken is one of the jobs to run (or none)
        self.action_space = Discrete(QUEUE_SIZE + 1)
        self.observation_space = Dict({
            "mask": Box(low=0.0, high=1.0, shape=(QUEUE_SIZE + 1,)),
            "obs": Box(low=0.0, high=np.inf, shape=(QUEUE_SIZE, 4 + ZONES))
        })

        self.workload = Workload(config["workload"])
        self.stride = int(self.workload.cores / ZONES)

    def build_observation(self):
        self.cluster.sort(key=lambda j:j.requested_time - j.run_time)
        self.queue.sort(key=lambda j:j.wait_time)

        self.available_cores = self.workload.cores
        if self.cluster:
            for j in self.cluster:
                self.available_cores -= j.cores
            cluster = np.concatenate([j.obs_cluster() for j in self.cluster])
            cluster = np.pad(cluster, (0, self.available_cores))[::self.stride]
        else:
            cluster = np.zeros(ZONES)

        mask = np.array([1 if j.cores <= self.available_cores else 0 for j in self.queue[:QUEUE_SIZE]] + [0] * (QUEUE_SIZE - len(self.queue)) + [1])

        if self.queue:
            obs = np.stack([np.concatenate([j.obs_queue(self.available_cores), cluster.squeeze()]) for j in self.queue])
            obs = obs[:QUEUE_SIZE] if len(self.queue) >= QUEUE_SIZE else np.pad(obs, ((0, QUEUE_SIZE - len(self.queue)), (0, 0)))
        else:
            obs = np.zeros((QUEUE_SIZE, 4 + ZONES))

        return {
            "mask": mask,
            "obs": obs
        }
    
    def reset(self):
        self.time = self.workload[0].submit_time
        self.next_job_index = 1

        # return obs
        # lists of jobs
        self.cluster = [self.workload[0]]
        self.queue = []

        self.hay_pending_jobs = True
        self.workload.reset()

        return self.build_observation()

    def step(self, action):
        # return obs, reward, done, info
        # reward: w1 * wait_time + w2 * cpu_utilization

        # step_pending also, based on jobs not yet in queue
        step_pending = self.workload[self.next_job_index].submit_time - self.time if self.hay_pending_jobs else self.workload.max_run_time
        # step in time
        if self.cluster:
            shortest_job = min(self.cluster, key=lambda j:j.actual_time - j.run_time)
            step_cluster = shortest_job.actual_time - shortest_job.run_time
        else:
            step_cluster = step_pending
        #step_benchmark = 600 - self.time // 600  # every 10 minutes

        if step_cluster < step_pending:  # and step_cluster < step_benchmark:
            self.time += step_cluster
              
            for j in self.cluster[:]:  # iterate over copy
                j.run_time += step_cluster
                if j.run_time >= j.actual_time:
                    self.cluster.remove(j)
            for j in self.queue:
                j.wait_time += step_cluster
        #elif step_pending < step_cluster:  # and step_pending < step_benchmark:
        else:
            self.time += step_pending
          
            for j in self.cluster:
                j.run_time += step_pending
            for j in self.queue:
                j.wait_time += step_pending
            
            if self.hay_pending_jobs:
                self.queue.append(self.workload[self.next_job_index])
                self.next_job_index += 1

        #else:
        #  self.time += step_benchmark
        #  self.utilization.append(self.available_cores / self.workload.cores)

        # schedule a job
        if action != QUEUE_SIZE:
            self.cluster.append(self.queue.pop(action))

        if self.next_job_index == len(self.workload):  # no more pending jobs!
            self.hay_pending_jobs = False
        done = (not self.hay_pending_jobs) and (not self.queue) and (not self.cluster)
        if done:
            reward = 0
            for j in self.workload:
                reward += j.wait_time
            return self.reset(), -reward / len(self.workload), True, {}
        return self.build_observation(), 0, False, {}
