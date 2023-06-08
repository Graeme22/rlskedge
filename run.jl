using ReinforcementLearning
using Flux
using Flux.Losses

include("./cluster.jl")
using .cluster

#save_dir = nothing
N_ENV = 2
UPDATE_FREQ = 32
env = MultiThreadEnv([ClusterEnv() for i in 1:N_ENV])
# (QUEUE_SIZE, ZONES + 4); QUEUE_SIZE
ns, na = size(state(env[1])), length(action_space(env[1]))
agent = Agent(
    policy = PPOPolicy(
        approximator = ActorCritic(
            actor = Chain(
                Dense(ns[end], 256, relu),
                Dense(256, na)
            ),
            critic = Chain(
                Dense(ns[end], 256, relu),
                Dense(256, 1)
            ),
            optimizer = ADAM(1e-3)
        ),# |> gpu,
        γ = 0.99f0,
        λ = 0.95f0,
        clip_range = 0.2f0,
        max_grad_norm = 0.5f0,
        n_epochs = 10,
        n_microbatches = 32,
        actor_loss_weight = 1.0f0,
        critic_loss_weight = 0.5f0,
        entropy_loss_weight = 0.001f0,
        update_freq = UPDATE_FREQ
    ),
    trajectory = MaskedPPOTrajectory(;
        capacity = UPDATE_FREQ,
        state = Array{Float32, 3} => (ns..., N_ENV),
        action = Vector{Int} => (N_ENV,),
        legal_actions_mask = Vector{Bool} => (na, N_ENV),
        action_log_prob = Vector{Float32} => (N_ENV,),
        reward = Vector{Float32} => (N_ENV,),
        terminal = Vector{Bool} => (N_ENV,)
    )
)

stop_condition = StopAfterEpisode(4)
hook = TotalBatchRewardPerEpisode(N_ENV)
experiment = Experiment(agent, env, stop_condition, hook, "rlskedge-ppo")

run(experiment)