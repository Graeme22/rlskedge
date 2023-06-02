module cluster

using ReinforcementLearning
using IntervalSets
using Random

include("job.jl")
using .job

export ClusterEnv

const QUEUE_SIZE, ZONES = 16, 8  # 128, 32

mutable struct ClusterEnv <: AbstractEnv
    workload::Workload
    time::Int
    next_job_index::Int
    reward::Union{Nothing, Float32}
    done::Bool
    cluster::Vector{Job}
    queue::Vector{Job}
    are_pending_jobs::Bool
    available_cores::Int
end

function RLBase.reset!(env::ClusterEnv)
    workload = Workload(rand(WORKLOADS))
    env.workload = workload
    env.time = workload.jobs[1].submit_time
    env.next_job_index = 2
    env.reward = nothing
    env.done = false
    env.cluster = [workload.jobs[1]]
    env.queue = []
    env.are_pending_jobs = true
    env.available_cores = workload.cores - workload.jobs[1].cores
end

function ClusterEnv()
    workload = Workload(rand(WORKLOADS))
    time = workload.jobs[1].submit_time
    next_job_index = 2
    reward = nothing
    done = false
    cluster = [workload.jobs[1]]
    queue = []
    are_pending_jobs = true
    available_cores = workload.cores - workload.jobs[1].cores

    ClusterEnv(workload, time, next_job_index, reward, done, cluster, queue, are_pending_jobs, available_cores)
end

RLBase.action_space(env::ClusterEnv) = Base.OneTo(QUEUE_SIZE + 1)
RLBase.state_space(env::ClusterEnv) = Space(fill(0..Inf, QUEUE_SIZE, ZONES + 4))
RLBase.reward(env::ClusterEnv) = env.reward
RLBase.is_terminated(env::ClusterEnv) = env.done

function RLBase.state(env::ClusterEnv)
    sort!(env.cluster, by = x -> x.requested_time - x.run_time)
    sort!(env.queue, by = x -> x.wait_time)
    
    if env.queue == []
        return fill(0, QUEUE_SIZE, ZONES + 4)
    end

    if env.cluster == []
        cluster = fill(0, ZONES)
    else
        cluster = [job_cluster(j, env.workload.max_run_time) for j in env.cluster]
        buffer = fill(0, env.workload.cores - 221)
        cluster = reduce(vcat, cluster)
        cluster = reduce(vcat, [cluster, buffer])
        stride = env.workload.cores ÷ ZONES
        cluster = cluster[1:stride:end]  # length is now ZONES
    end

    jobs = [job_queue(j, env.workload.max_run_time, env.workload.cores, env.available_cores) for j in env.queue]
    queue = reduce(hcat, jobs)'

    n_queued_jobs = size(queue)[1] < QUEUE_SIZE ? size(queue)[1] : QUEUE_SIZE
    cluster = repeat(cluster, 1, n_queued_jobs)'

    obs = hcat(queue, cluster)
    padding = fill(0, QUEUE_SIZE - n_queued_jobs, ZONES + 4)
    obs = vcat(obs, padding)

    if obs ∉ RLBase.state_space(env)
        println("ERROR: obs ∉ RLBase.state_space(env)")
        println(env.queue)
        println(env.cluster)
    end
    obs
end

RLBase.legal_action_space(env::ClusterEnv) = findall(legal_action_space_mask(env))

function RLBase.legal_action_space_mask(env::ClusterEnv)
    if env.queue == []
        mask = vcat(fill(0, QUEUE_SIZE), [1])
    elseif length(env.queue) < QUEUE_SIZE
        queue = [j.cores <= env.available_cores for j in env.queue[1:length(env.queue)]]
        buffer = fill(0, QUEUE_SIZE - length(env.queue))
        mask = vcat(queue, buffer, [1])
    else
        queue = [j.cores <= env.available_cores for j in env.queue[1:QUEUE_SIZE]]
        mask = vcat(queue, [1])
    end

    mask .== 1
end

function (env::ClusterEnv)(action)
    # based on jobs not yet in queue i.e. pending jobs
    step_pending = env.are_pending_jobs ? env.workload.jobs[env.next_job_index].submit_time - env.time : env.workload.max_run_time

    if env.cluster != []
        shortest_job_index = findmin(j -> j.run_time - j.simulated_run_time, env.cluster)[2]
        shortest_job = env.cluster[shortest_job_index]
        step_cluster = shortest_job.run_time - shortest_job.simulated_run_time
    else
        step_cluster = step_pending
    end
    #step_benchmark = 600 - self.time // 600  # every 10 minutes

    # smallest step is an existing job
    if step_cluster < step_pending  # and step_cluster < step_benchmark:
        env.time += step_cluster
            
        to_remove = []
        for i in 1:length(env.cluster)
            env.cluster[i].simulated_run_time += step_cluster
            if env.cluster[i].simulated_run_time >= env.cluster[i].run_time
                env.available_cores += env.cluster[i].cores
                push!(to_remove, i)
            end
        end
        deleteat!(env.cluster, to_remove)

        for j in env.queue
            j.simulated_wait_time += step_cluster
        end
    
    #elif step_pending < step_cluster:  # and step_pending < step_benchmark:
    else
        env.time += step_pending
        
        for j in env.cluster
            j.simulated_run_time += step_pending
        end
        for j in env.queue
            j.simulated_wait_time += step_pending
        end
        
        if env.are_pending_jobs
            push!(env.queue, env.workload.jobs[env.next_job_index])
            env.next_job_index += 1
        end
    end
    
    #else:
    #  self.time += step_benchmark
    #  self.utilization.append(self.available_cores / self.workload.cores)
    
    # reward: w1 * wait_time + w2 * cpu_utilization
    
    # schedule a job
    if action != QUEUE_SIZE + 1  # noop
        j = popat!(env.queue, action)
        env.available_cores -= j.cores
        push!(env.cluster, j)
    end
    
    if env.next_job_index > length(env.workload.jobs)  # no more pending jobs!
        env.are_pending_jobs = false
    end
    done = (!env.are_pending_jobs) && env.queue == [] && env.cluster == []
    if done
        reward = 0
        for j in env.workload.jobs
            reward += j.simulated_wait_time
        end
        env.reward = -reward / length(env.workload.jobs)
        env.done = true
    end
end

RLBase.StateStyle(::ClusterEnv) = Observation{Array{2}}()
RLBase.ActionStyle(::ClusterEnv) = FULL_ACTION_SET
RLBase.RewardStyle(::ClusterEnv) = TERMINAL_REWARD
RLBase.ChanceStyle(::ClusterEnv) = DETERMINISTIC

end
