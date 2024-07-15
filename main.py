import os
import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger

#
# REFER TO THE GUIDE FOR INFORMATION ON ALL VALUES.
# MINOR ANNOTATIONS WILL BE MADE HERE AND THERE IN THIS FILE WITH COMMENTS.
#
#
#

g_combined_reward = None # type: LogCombinedReward

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score] + [g_combined_reward.prev_rewards]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps) -> None:
        global g_combined_reward
        avg_rewards = np.zeros(len(g_combined_reward.reward_functions))
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
            avg_rewards += metric_array[3]
        avg_rewards /= len(collected_metrics)
        avg_linvel /= len(collected_metrics)
        num_days_played = cumulative_timesteps / (120/8) /60 /60 / 24
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps,
                  "Days":num_days_played,
                  "Years played":num_days_played/365}
        for i in range(len(g_combined_reward.reward_functions)):
            report["reward/ " + g_combined_reward.reward_functions[i].__class__.__name__] = avg_rewards[i]
        wandb_run.log(report)

def get_most_recent_checkpoint() -> str:
    checkpoint_load_dir = "data/checkpoints/"
    checkpoint_load_dir += str(
        max(os.listdir(checkpoint_load_dir), key=lambda d: int(d.split("-")[-1])))
    checkpoint_load_dir += "/"
    checkpoint_load_dir += str(
        max(os.listdir(checkpoint_load_dir), key=lambda d: int(d)))
    return checkpoint_load_dir

def build_rocketsim_env(): # build our environment
    import rlgym_sim
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityBallToGoalReward, \
        EventReward, TouchBallReward, FaceBallReward #add rewards here if you'd like to import more from the default selection.
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.state_setters.random_state import RandomState
    from rlgym_sim.utils.state_setters.default_state import DefaultState
    from necto_act import NectoAction # I use Necto Action Parser because it's got 90 actions to pick from. Smaller action space means your bot can start to focus on movement and stuff faster.
                                      # Even though your bot doesn't know what focus is.
    from custom_rewards import SpeedTowardBallReward, TouchBallRewardScaledByHitForce, SpeedflipKickoffReward, AerialDistanceReward, PlayerOnWallReward, LavaFloorReward, InAirReward
    from custom_state_setters import WeightedSampleSetter, WallPracticeState, TeamSizeSetter
    from custom_combinedreward import LogCombinedReward
    spawn_opponents = True # Whether you want opponents or not, set to False if you're practicing hyperspecific scenarios. The opponent is your own bot, so it plays against itself. Used to be called self_play in the old days.
    team_size = 1 # How many bots per team.
    game_tick_rate = 120
    tick_skip = 8 # How long we hold an action before taking a new action, in ticks.
    timeout_seconds = 15
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = NectoAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()] # What conditions terminate the current episode.

    aggression_bias = 0.2
    goal_reward = 1
    concede_reward = -goal_reward * (1 - aggression_bias)

    reward_fn = LogCombinedReward.from_zipped(
                (EventReward(team_goal=goal_reward, 
                             concede=concede_reward), 20), # Different event reward.
                (VelocityBallToGoalReward(), 10),
                (TouchBallRewardScaledByHitForce(), 1),
                (SpeedTowardBallReward(), 1), # Move towards the ball!
                (FaceBallReward(), 0.05), # Make sure we don't start driving backward at the ball
                (InAirReward(), 0.05) # Make sure we don't forget how to jump
    )
    global g_combined_reward
    g_combined_reward = reward_fn
    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)
    
    state_setter = WeightedSampleSetter.from_zipped(
        (RandomState(True, True, False), 0.5),
        (DefaultState(), 0.5),
        (WallPracticeState(), 0.2),
    ) 
    # Pre-set according to the guide, ordered in random ball velocity, random car velocity and whether the cars will be on the ground.

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)
    
    import renderer.rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    n_proc = 34
    print(f"Initializing {n_proc} instances.")
    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    try:
        checkpoint_load_dir = get_most_recent_checkpoint()
        print(f"Loading checkpoint: {checkpoint_load_dir}")
    except:
        print("checkpoint load dir not found.")
        checkpoint_load_dir = None
    learner = Learner(build_rocketsim_env,
                      checkpoint_load_folder=checkpoint_load_dir, # What folder you want to load your bot checkpoint from, not automatically the latest. So we grab the latest with the above function.
                      n_proc=n_proc, # Instance count
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger, # Metric logger instance, what gets logged to Weights And Biases.
                      ppo_batch_size=50_000, # Make this the same value as ts_per_iteration.
                      ts_per_iteration=50_000, 
                      exp_buffer_size=150_000,
                      policy_lr=2e-4, # change these according to the guide, KEEP THEM THE SAME UNLESS YOU KNOW WHAT YOU'RE DOING.
                      critic_lr=2e-4, # 7e-4 for a brand new bot, 
                                      # 3e-4 when your bot starts chasing the ball and hitting it, 
                                      # 2e-4 when it starts trying to score
                                      # 1e-4 for advanced outplay mechs.
                      # policy_layer_sizes=(2048, 1024, 1024, 512), future plans
                      # critic_layer_sizes=(2048, 1024, 1024, 512), future plans
                      ppo_minibatch_size=50_000,
                      ppo_ent_coef=0.001, # golden value is near 0.01.
                      ppo_epochs=1,
                      # render=True, # set to True if you want to see your bot play, don't keep it on though as it slows your learning.
                      # render_delay=8/120,
                      standardize_returns=True,
                      add_unix_timestamp=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=1e69, # how many timesteps until it's supposed to stop learning, currently 1 billion by default.
                      log_to_wandb=True, # True if you want to log to Weights And Biases, keep False otherwise.
                      wandb_run_name="Apollo-X", # Name of your Weights And Biases run.
                      wandb_project_name="Apollo", # Name of your Weights And Biases project.
                      wandb_group_name="Dionysus-Family" # Name of the Weights And Biases project group.
                      )
    build_rocketsim_env()
    learner.learn()
