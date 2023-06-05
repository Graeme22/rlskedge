module cluster

using ReinforcementLearning
using IntervalSets
using Random
using Statistics

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
    utilization::Vector{Float32}
    avg_wait_time::Float32
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
    env.utilization = []
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
    utilization = []
    avg_wait_time = sum(j -> j.wait_time, workload.jobs) / length(workload.jobs)

    ClusterEnv(workload, time, next_job_index, reward, done, cluster, queue, are_pending_jobs, available_cores, utilization, avg_wait_time)
end

RLBase.action_space(env::ClusterEnv) = Base.OneTo(QUEUE_SIZE + 1)
RLBase.state_space(env::ClusterEnv) = Space(fill(0..Inf, QUEUE_SIZE, ZONES + 4))
RLBase.reward(env::ClusterEnv) = env.reward
RLBase.is_terminated(env::ClusterEnv) = env.done

function RLBase.state(env::ClusterEnv)
    if env.queue == []
        return fill(0, QUEUE_SIZE, ZONES + 4)
    end

    if env.cluster == []
        cluster = fill(0, ZONES)
    else
        cluster = [job_cluster(j, env.workload.max_run_time) for j in env.cluster]
        buffer = fill(0, env.available_cores)
        cluster = reduce(vcat, cluster)
        cluster = reduce(vcat, [cluster, buffer])
        stride = cld(env.workload.cores, ZONES)
        cluster = cluster[1:stride:end]  # length is now ZONES
    end

    if length(env.queue) > QUEUE_SIZE
        jobs = [job_queue(j, env.workload.max_run_time, env.workload.cores, env.available_cores) for j in env.queue[1:QUEUE_SIZE]]
    else
        jobs = [job_queue(j, env.workload.max_run_time, env.workload.cores, env.available_cores) for j in env.queue]
    end
    queue = reduce(hcat, jobs)'

    n_queued_jobs = size(queue)[1]
    cluster = repeat(cluster, 1, n_queued_jobs)'
    
    obs = hcat(queue, cluster)
    padding = fill(0, QUEUE_SIZE - n_queued_jobs, ZONES + 4)
    obs = vcat(obs, padding)

    obs
end

RLBase.legal_action_space(env::ClusterEnv) = findall(legal_action_space_mask(env))

function RLBase.legal_action_space_mask(env::ClusterEnv)
    if env.queue == []
        mask = vcat(fill(0, QUEUE_SIZE), [1])
    elseif length(env.queue) < QUEUE_SIZE
        queue = [j.cores <= env.available_cores for j in env.queue]
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
    step_benchmark = 3600 - env.time ÷ 3600  # every hour

    if env.cluster != []
        shortest_job_index = findmin(j -> j.run_time - j.simulated_run_time, env.cluster)[2]
        shortest_job = env.cluster[shortest_job_index]
        step_cluster = shortest_job.run_time - shortest_job.simulated_run_time
    else
        step_cluster = Inf
    end

    step = min(step_pending, step_benchmark, step_cluster)
    env.time += step
    for j in env.cluster
        j.simulated_run_time += step
    end
    for j in env.queue
        j.simulated_wait_time += step
    end

    if step_cluster == step
        to_remove = []
        for i in 1:length(env.cluster)
            if env.cluster[i].simulated_run_time >= env.cluster[i].run_time
                env.available_cores += env.cluster[i].cores
                push!(to_remove, i)
            end
        end
        deleteat!(env.cluster, to_remove)
    end
    if step_pending == step
        if env.are_pending_jobs
            push!(env.queue, env.workload.jobs[env.next_job_index])
            env.next_job_index += 1
        end
    end
    if step_benchmark == step
        push!(env.utilization, env.available_cores / env.workload.cores)
    end

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
        # reward: w1 * wait_time + w2 * cpu_utilization
        # utilization, wait time for (F1, SJF, FCFS)
        reward = 0
        for j in env.workload.jobs
            reward += j.simulated_wait_time
        end
        avg_wait_time = reward / length(env.workload.jobs) / env.avg_wait_time
        avg_utilization = mean(env.utilization)
        # todo: make this reliable
        env.reward = avg_utilization - avg_wait_time
        env.done = true
    else
        sort!(env.cluster, by = x -> x.run_time - x.requested_time)
        sort!(env.queue, by = x -> -x.wait_time)
    end
end

RLBase.NumAgentStyle(::ClusterEnv) = SINGLE_AGENT
RLBase.StateStyle(::ClusterEnv) = Observation{Array{2}}()
RLBase.RewardStyle(::ClusterEnv) = TERMINAL_REWARD
RLBase.ActionStyle(::ClusterEnv) = FULL_ACTION_SET
RLBase.NumAgentStyle(::ClusterEnv) = SINGLE_AGENT
RLBase.UtilityStyle(::MontyHallEnv) = GENERAL_SUM
RLBase.InformationStyle(::ClusterEnv) = IMPERFECT_INFORMATION
RLBase.ChanceStyle(::ClusterEnv) = STOCHASTIC

end
