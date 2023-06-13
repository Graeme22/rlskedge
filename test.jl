using ReinforcementLearning
using Flux
using Flux.Losses

include("./cluster.jl")
using .cluster

# repeat for random, fcfs, sjf
for i in 1:19
    env = ClusterEnv(i)
    hook = DoEveryNStep(;n=100000) do t, agent, env
        println("Job ", env.next_job_index, "/", length(env.workload.jobs))
    end
    stop = StopAfterEpisode(1)
    run(RandomPolicy(), env, stop, hook)
    println("Metrics $i: ", env.metrics)
end
