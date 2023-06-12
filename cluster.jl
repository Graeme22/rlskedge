module cluster

using ReinforcementLearning
using IntervalSets
using Statistics
using Random

include("job.jl")
using .job

export ClusterEnv, QUEUE_SIZE, ZONES

const QUEUE_SIZE, ZONES = 128, 32

struct Metrics
    avg_bounded_slowdown::Float32
    avg_wait_time::Float32
    max_wait_time::Int
    avg_utilization::Float32
end

mutable struct ClusterEnv <: AbstractEnv
    workload::Workload
    time::Int
    next_job_index::Int
    reward::Float32
    done::Bool
    cluster::Vector{Job}
    queue::Vector{Job}
    are_pending_jobs::Bool
    available_cores::Int
    utilization::Vector{Float32}
    metrics::Union{Nothing, Metrics}
end

function RLBase.reset!(env::ClusterEnv)
    workload = Workload(rand(WORKLOADS))
    env.workload = workload
    env.time = workload.jobs[1].submit_time
    env.next_job_index = 1
    env.reward = 0
    env.done = false
    env.cluster = []
    env.queue = [workload.jobs[1]]
    env.are_pending_jobs = true
    env.available_cores = workload.cores
    env.utilization = []
    env.metrics = nothing
end

function ClusterEnv()
    workload = Workload(rand(WORKLOADS))
    time = workload.jobs[1].submit_time
    next_job_index = 1
    reward = 0
    done = false
    cluster = []
    queue = [workload.jobs[1]]
    are_pending_jobs = true
    available_cores = workload.cores
    utilization = []
    metrics = nothing

    ClusterEnv(workload, time, next_job_index, reward, done, cluster, queue, are_pending_jobs, available_cores, utilization, metrics)
end

function ClusterEnv(index::Int)
    workload = Workload(WORKLOADS[index])
    time = workload.jobs[1].submit_time
    next_job_index = 1
    reward = 0
    done = false
    cluster = []
    queue = [workload.jobs[1]]
    are_pending_jobs = true
    available_cores = workload.cores
    utilization = []
    metrics = nothing

    ClusterEnv(workload, time, next_job_index, reward, done, cluster, queue, are_pending_jobs, available_cores, utilization, metrics)
end

RLBase.action_space(env::ClusterEnv) = Base.OneTo(QUEUE_SIZE)
RLBase.state_space(env::ClusterEnv) = Space(fill(0..Inf, ZONES + 4, QUEUE_SIZE))
RLBase.reward(env::ClusterEnv) = env.reward
RLBase.is_terminated(env::ClusterEnv) = env.done

function RLBase.state(env::ClusterEnv)
    if env.queue == []
        return fill(0, ZONES + 4, QUEUE_SIZE)
    end

    if env.cluster == []
        cluster = fill(0, ZONES)
    else
        cluster = [job_cluster(j, env.workload.max_run_time) for j in env.cluster]
        buffer = fill(0, env.available_cores)
        cluster = reduce(vcat, cluster)
        cluster = reduce(vcat, [cluster, buffer])
        stride = env.workload.cores / ZONES
        cluster = [cluster[round(Int, i * stride)] for i in 1:ZONES]
    end

    n_queued_jobs = length(env.queue) < QUEUE_SIZE ? length(env.queue) : QUEUE_SIZE
    jobs = [job_queue(j, env.workload.max_run_time, env.workload.cores, env.available_cores) for j in env.queue[1:n_queued_jobs]]
    queue = reduce(hcat, jobs)
    cluster = repeat(cluster, 1, n_queued_jobs)
    
    obs = vcat(queue, cluster)
    padding = fill(0, ZONES + 4, QUEUE_SIZE - n_queued_jobs)
    obs = hcat(obs, padding)

    obs
end

RLBase.legal_action_space(env::ClusterEnv) = findall(legal_action_space_mask(env))

function RLBase.legal_action_space_mask(env::ClusterEnv)
    mask = fill(0, QUEUE_SIZE)
    n_queued_jobs = length(env.queue) < QUEUE_SIZE ? length(env.queue) : QUEUE_SIZE
    for i in 1:n_queued_jobs
        mask[i] = 1
    end

    mask .== 1
end

function (env::ClusterEnv)(action)
    # FCFS
    action = 1
    # SJF
    """
    action = 1
    n_queued_jobs = length(env.queue) < QUEUE_SIZE ? length(env.queue) : QUEUE_SIZE
    for i in 2:n_queued_jobs
        if env.queue[i].requested_time < env.queue[action].requested_time
            action = i
        end
    end
    #"""
    # chosen job
    job_to_schedule = popat!(env.queue, action)
    scheduled = false
    # loops until chosen job can be scheduled
    while !scheduled || legal_action_space(env) == []
        # schedule it if we can
        if !scheduled && job_to_schedule.cores <= env.available_cores
            env.available_cores -= job_to_schedule.cores
            push!(env.cluster, job_to_schedule)
            scheduled = true
        # otherwise, backfill jobs
        else
            min_time_to_completion = Inf
            for j in env.cluster
                if j.requested_time - j.simulated_run_time < min_time_to_completion
                    min_time_to_completion = j.run_time - j.simulated_run_time
                end
            end
            n_queued_jobs = length(env.queue) < QUEUE_SIZE ? length(env.queue) : QUEUE_SIZE
            to_remove = []
            # TODO: sort using logits for policy
            #sorted = sort(env.queue, by = _ -> rand()) # randomize order until we have logits
            # FCFS
            sorted = sort(env.queue, by = j -> j.submit_time)
            # SJF
            #sorted = sort(env.queue, by = j -> j.requested_time)
            for i in 1:n_queued_jobs
                if sorted[i].cores <= env.available_cores && sorted[i].requested_time <= min_time_to_completion
                    push!(to_remove, sorted[i])
                    env.available_cores -= sorted[i].cores
                    push!(env.cluster, sorted[i])
                end
            end
            filter!(j -> j ∉ to_remove, env.queue)
        end

        # based on jobs not yet in queue i.e. pending jobs
        step_pending = env.are_pending_jobs ? env.workload.jobs[env.next_job_index].submit_time - env.time : env.workload.max_run_time
        step_benchmark = 3600 - env.time % 3600  # every hour

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
        if step_pending == step && env.are_pending_jobs
            push!(env.queue, env.workload.jobs[env.next_job_index])
            env.next_job_index += 1
        end
        if step_benchmark == step
            push!(env.utilization, env.available_cores / env.workload.cores)
        end
        
        # check for termination
        if env.next_job_index > length(env.workload.jobs)  # no more pending jobs!
            env.are_pending_jobs = false
        end
        # we've finished!
        if (!env.are_pending_jobs) && env.queue == [] && env.cluster == []
            # calculate trace metrics
            avg_wait_time = 0
            max_wait_time = -Inf
            for j in env.workload.jobs
                avg_wait_time += j.simulated_wait_time
                if j.simulated_wait_time > max_wait_time
                    max_wait_time = j.simulated_wait_time
                end
            end
            avg_wait_time /= length(env.workload.jobs)
            avg_utilization = mean(env.utilization)
            bslds = [max((j.simulated_wait_time + j.simulated_run_time) / max(j.simulated_run_time, 10), 1) for j in env.workload.jobs]
            sum_bslds = +(bslds...)
            avg_bsld = sum_bslds / length(env.workload.jobs)
            # currently: negative average bounded slowdown
            env.reward = -avg_bsld
            env.done = true
            env.metrics = Metrics(avg_bsld, avg_wait_time, max_wait_time, avg_utilization)
            break
        else
            sort!(env.cluster, by = j -> j.requested_time - j.simulated_run_time, rev = true)
        end
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
