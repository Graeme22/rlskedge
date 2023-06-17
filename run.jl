using ReinforcementLearning
using Flux
using Flux.Losses
using Flux: params
using BSON
using Logging
using TensorBoardLogger
using Statistics

include("./cluster.jl")
using .cluster

save_dir = "save"
N_ENV = 8
TRAJECTORY_SIZE = 1024
env = MultiThreadEnv([ClusterEnv() for i in 1:N_ENV])

shared_network = Chain(
    x -> reshape(x, ZONES + 4, QUEUE_SIZE, :),
    Dense(ZONES + 4, 512, relu),
    Dense(512, 1),
    x -> reshape(x, QUEUE_SIZE, :)  # squeeze
)

agent = Agent(
    policy = PPOPolicy(
        approximator = ActorCritic(
            actor = shared_network,
            critic = Chain(
                shared_network,
                Dense(QUEUE_SIZE, 256, relu),
                Dense(256, 1),
                x -> reshape(x, :)  # squeeze
            ),
            optimizer = Adam(1e-5)
        ) |> gpu,
        γ = 0.99f0,
        λ = 0.95f0,
        clip_range = 0.2f0,
        max_grad_norm = 0.5f0,
        n_epochs = 50,
        n_microbatches = 32,
        actor_loss_weight = 1.0f0,
        critic_loss_weight = 0.5f0,
        entropy_loss_weight = 0.001f0,
        update_freq = TRAJECTORY_SIZE
    ),
    trajectory = MaskedPPOTrajectory(;
        capacity = TRAJECTORY_SIZE,
        state = Array{Float32, 3} => (ZONES + 4, QUEUE_SIZE, N_ENV),
        action = Vector{Int} => (N_ENV,),
        legal_actions_mask = Vector{Bool} => (QUEUE_SIZE, N_ENV),
        action_log_prob = Vector{Float32} => (N_ENV,),
        reward = Vector{Float32} => (N_ENV,),
        terminal = Vector{Bool} => (N_ENV,)
    )
)

Base.@kwdef mutable struct TimeCostPerEpisode <: AbstractHook
    t::UInt64 = time_ns()
    time_costs::Vector{UInt64} = []
end

logger = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)

rwd = TotalBatchRewardPerEpisode(N_ENV)
param_dir = mktempdir()
stop_condition = StopAfterStep(1000000)
hook = ComposedHook(
    rwd,
    BatchStepsPerEpisode(N_ENV),
    TimeCostPerEpisode(),
    DoEveryNStep(;n=1000000) do t, agent, env
        ps = params(agent.policy)
	f = joinpath(param_dir, "params.bson")
	BSON.@save f ps
	println("parameters saved to $f")
    end,
    DoEveryNStep(;n=100) do t, agent, env
	p = agent.policy
	with_logger(logger) do
	    @info "training" loss = mean(p.loss) actor_loss = mean(p.actor_loss) critic_loss = mean(p.critic_loss) entropy_loss = mean(p.entropy_loss) norm = mean(p.norm)
	end
    end,
    DoEveryNStep(;n=1000) do t, agent, env
	with_logger(logger) do
	    rw = [rwd.rewards[i][end] for i in 1:length(env) if is_terminated(env[i])]
	    if length(rw) > 0
	        @info "training" rewards = mean(rewards)
	    end
	end
    end	
)

run(agent, env, stop_condition, hook)
