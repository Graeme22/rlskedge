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
UPDATE_FREQ = 32
env = MultiThreadEnv([ClusterEnv() for i in 1:N_ENV])

agent = Agent(
    policy = PPOPolicy(
        approximator = ActorCritic(
            actor = Chain(
                x -> reshape(x, ZONES + 4, QUEUE_SIZE, :),
                Dense(ZONES + 4, 1024, relu),
                Dense(1024, 1024, relu),
                Dense(1024, 1),
                x -> reshape(x, QUEUE_SIZE, :)  # squeeze
            ),
            critic = Chain(
                x -> reshape(x, ZONES + 4, QUEUE_SIZE, :),
                Dense(ZONES + 4, 256, relu),
                Dense(256, 256, relu),
                Dense(256, 1),
                x -> reshape(x, QUEUE_SIZE, :),  # squeeze
                Dense(QUEUE_SIZE, 256, relu),
                Dense(256, 256, relu),
                Dense(256, 1),
                x -> reshape(x, :)  # squeeze
            ),
            optimizer = ADAM(1e-3)
        ) |> gpu,
        γ = 0.99f0,
        λ = 0.95f0,
        clip_range = 0.2f0,
        max_grad_norm = 0.5f0,
        n_epochs = 4,
        n_microbatches = 4,
        actor_loss_weight = 1.0f0,
        critic_loss_weight = 0.5f0,
        entropy_loss_weight = 0.001f0,
        update_freq = UPDATE_FREQ
    ),
    trajectory = MaskedPPOTrajectory(;
        capacity = UPDATE_FREQ,
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

rewards = TotalBatchRewardPerEpisode(N_ENV)
param_dir = mktempdir()
stop_condition = StopAfterStep(1000000)
hook = ComposedHook(
    rewards,
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
	    rw = [rewards.rewards[i][end] for i in 1:length(env) if is_terminated(env[i])]
	    if length(rw) > 0
	        @info "training" rewards = mean(rewards)
	    end
	end
    end	
)

run(agent, env, stop_condition, hook)
