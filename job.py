import numpy as np
import re
from torch.utils.data import Dataset


class Job:
    def __init__(self, workload, line="0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0"):
        self.workload = workload
        s_array = re.split("\\s+", line.strip())
        self.job_id = int(s_array[0])
        self.submit_time = int(s_array[1])
        self.recorded_wait_time = int(s_array[2])
        self.wait_time = 0
        self.actual_time = int(s_array[3])
        self.run_time = 0
        self.number_of_allocated_processors = int(s_array[4])
        self.request_number_of_processors = int(s_array[7])
        self.cores = max(self.number_of_allocated_processors, self.request_number_of_processors)
        self.requested_time = int(s_array[8])
        if self.requested_time == -1:
            self.requested_time = self.actual_time

    def __eq__(self, other):
        return self.job_id == other.job_id

    def __lt__(self, other):
        return self.job_id < other.job_id

    def obs_queue(self, available_cores):
        # job characteristics are: wait time, requested time, # of CPUs, is currently schedulable.
        return np.array([self.wait_time / self.workload.max_run_time, self.requested_time / self.workload.max_run_time, self.cores / self.workload.cores, float(available_cores >= self.cores)])

    def obs_cluster(self):
        return np.full((self.cores,), (self.requested_time - self.run_time) / self.workload.max_run_time)


class Workload(Dataset):
    def __init__(self, path):
        self.all_jobs = []

        with open(path) as fp:
            for line in fp:
                if line.startswith(";"):
                    if line.startswith("; MaxNodes:"):
                        self.cores = int(line.split(":")[1].strip())
                    if line.startswith("; MaxRuntime:"):
                        self.max_run_time = int(line.split(":")[1].strip())
                    continue

                if not self.max_run_time or not self.cores:
                    raise Exception("Improper workload format! Comments must include MaxNodes and MaxRuntime properties.")
                j = Job(self, line)
                # filter those illegal data whose runtime <= 0
                if j.actual_time > 0:
                    self.all_jobs.append(j)

        self.all_jobs.sort(key=lambda job: job.job_id)

    def __len__(self):
        return len(self.all_jobs)

    def __getitem__(self, item):
        return self.all_jobs[item]

    def reset(self):
        for j in self.all_jobs:
            j.wait_time = 0
            j.run_time = 0
