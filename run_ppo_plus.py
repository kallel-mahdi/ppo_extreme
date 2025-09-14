# %%

import argparse
import logging
import os

import jax.numpy as jnp


import os
from collections import deque
from functools import partial

import gymnasium as gym
import jax
import numpy as np
import tqdm
import wandb
from jax import config

from jaxrl_m.dataset import ActorReplayBuffer, ReplayBuffer
from jaxrl_m.rollout import (rollout_policy, rollout_policy2)
from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb
from jaxrl_m.ppo_plus_off import SuperPPOConfig, create_learner
from jaxrl_m.utils import *
from jaxrl_m.normalize import *

import random




# Set env variables
os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
os.environ["WANDB__SERVICE_WAIT"] = str(1800)
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['XLA_FLAGS']='--xla_gpu_deterministic_ops=true'
# Enable highest precision in JAX
#config.update('jax_enable_x64', True)  # Enable 64-bit precision
#jax.config.update('jax_default_matmul_precision', 'highest')  # Use highest precision for matrix multiplications



##############################
parser = argparse.ArgumentParser()

parser.add_argument('--seed',type=int,default=21) 

parser.add_argument('--algo_name', type=str, default='superppo', help='the name of the RL algorithm')
parser.add_argument('--project_name',type=str,default="single_exp_off") 
parser.add_argument('--env_name',type=str,default="Ant-v5") 
parser.add_argument('--max_steps',type=int,default=None) 
parser.add_argument('--max_episode_steps',type=int,default=1000) 
parser.add_argument('--gamma',type=float,default=0.99)
parser.add_argument('--entropy_coeff',type=float,default=0.5) 

parser.add_argument('--num_critics',type=int,default=2)
parser.add_argument('--hidden_dims',type=int,default=256) 
parser.add_argument('--momentum',type=float,default=0.9) 
parser.add_argument('--b2',type=float,default=0.999) 
parser.add_argument('--temperature',type=float,default=1.) 


parser.add_argument('--on_policy_critic',type=str2bool,default=False)
parser.add_argument('--on_policy_actor',type=str2bool,default=False)
parser.add_argument('--min_target',type=str2bool,default=False)
parser.add_argument('--use_layer_norm',type=str2bool,default=True)

parser.add_argument('--clipping_ratio',type=float,default=0.25) 
parser.add_argument('--gae_lambda',type=float,default=0.) 

parser.add_argument('--episode_based',type=str2bool,default=False) 
parser.add_argument('--buffer_size',type=int,default=50_000) 
parser.add_argument('--policy_steps',type=int,default=5000) 
parser.add_argument('--num_epochs',type=int,default=25) 
parser.add_argument('--activation_fn',type=str,default='silu')
parser.add_argument('--stable_scheme',type=str2bool,default=True)
parser.add_argument('--bound_actions',type=str2bool,default=True)
parser.add_argument('--optimizer',type=str,default='sgd', choices=['adam','rmsprop', 'sgd','adamw'])
parser.add_argument('--spo_loss',type=str2bool,default=True)

args = parser.parse_args()
print(args)


from copy import deepcopy

random.seed(args.seed)
np.random.seed(args.seed)
jax_rng = jax.random.PRNGKey(args.seed)
jax.config.update("jax_default_matmul_precision", "highest")

def train(args):
    
    
        
    
    if args.on_policy_critic: args.buffer_size = args.policy_steps
    max_steps = get_max_steps_for_env(args.env_name) if args.max_steps is None else args.max_steps
    
    log_interval = 20000
    n_grads = 0

    wandb_config = {
        'project': args.project_name,
        'name':None,
        'hyperparam_dict':args.__dict__,
        }
    wandb_run = setup_wandb(**wandb_config)
    
    env,eval_env = create_environments(args.env_name)
    
  
    

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        truncateds = 0.0,
        next_observations=env.observation_space.sample(),
        pre_actions = env.action_space.sample(),
        discounts=1.0,
        log_probs=0.,
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(args.buffer_size))
    actor_buffer = ActorReplayBuffer.create(example_transition, size=args.policy_steps)

    config = SuperPPOConfig.from_args(args)

    agent = create_learner(  config=config,
                            observations=example_transition['observations'][None],
                            actions=example_transition['actions'][None]
                    )

    exploration_metrics = dict()
    exploration_rng = jax.random.PRNGKey(0)
    i = 0
    unlogged_steps,cached_steps = 0,0
    
    rollout_fn = rollout_policy if args.episode_based else rollout_policy2

    with tqdm.tqdm(total=max_steps) as pbar:
        
        while (i < max_steps):
                
                logging.debug('policy rollout')
                if args.on_policy_critic: replay_buffer = replay_buffer.reset()
                replay_buffer,actor_buffer,policy_return,undisc_policy_return,num_steps = rollout_fn(
                                                                        agent,env,exploration_rng,
                                                                        replay_buffer,actor_buffer,eval=False,
                                                                        discount = args.gamma,max_steps=args.policy_steps)
                         
                
                unlogged_steps += num_steps
                cached_steps += num_steps
                i+=num_steps
                pbar.update(int(num_steps))
                
                
                for _ in range(args.num_epochs):
            
                    ### Update critics ###:
                    logging.debug('update critics')
                    transitions = replay_buffer.get_all()
                    agent = agent.update_critics_seq(transitions)
                
                
                for _ in range(args.num_epochs):
                           
                    ### Update actor ###
                    if args.on_policy_actor:
                        actor_batch = actor_buffer.get_all()
                    else:
                        actor_batch = replay_buffer.get_all()
                    
                    agent,actor_update_info = agent.update_actor_seq(actor_batch)
                    critic_update_info = {}
                
                update_info = {**critic_update_info, **actor_update_info}
                agent = agent.replace(old_actor_params=deepcopy(agent.actor.params),old_temp_params=deepcopy(agent.temp.params))
                n_grads += args.num_epochs * 2  # epochs for critics + epochs for actor
                
                ### Log training info ###
                exploration_metrics = {f'exploration/disc_return': policy_return}
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                train_metrics['training/undisc_return'] = undisc_policy_return
                
                # Combine all metrics into a single log call to avoid step conflicts
                all_metrics = {**train_metrics, **exploration_metrics}
                wandb.log(all_metrics, step=int(i))
            
                ### Log evaluation info ###
                
                if unlogged_steps >= log_interval:
                    
                    _,_,policy_return,undisc_policy_return,num_steps = rollout_policy(
                                                                    agent,eval_env,exploration_rng,
                                                                    None,None,eval=True,
                                                                    discount = args.gamma,max_rollouts=10)
                    eval_metrics = {"policy_return": policy_return,"undisc_policy_return": undisc_policy_return}
                    print(eval_metrics)
                    
                    eval_metrics = {f'evaluation/{k}': v for k, v in eval_metrics.items()}
                    eval_metrics['n_grads']=int(n_grads)
                    
                    wandb.log(eval_metrics, step=int(i),commit=True)
                    unlogged_steps = 0
        
    wandb_run.finish()

train(args)
#%%
