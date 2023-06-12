using ReinforcementLearning
using Flux
using Flux.Losses

include("./cluster.jl")
using .cluster

for i in 1:21
    env = ClusterEnv(i)
    run(RandomPolicy(), env, StopAfterEpisode(1))
    println("Env $i FCFS metrics: ", env.metrics)
end