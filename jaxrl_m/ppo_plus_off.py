import jax.random
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.training.train_state import TrainState
from flax.struct import field as flax_field
from jaxrl_m.networks import OriginalCritic, Policy,ensemblize
from jaxrl_m.typing import *
from dataclasses import dataclass, field
from functools import partial

@dataclass
class NetworkConfig:
    """Network architecture configuration."""
    hidden_dims: int = 256
    activation_fn: str = 'tanh'
    use_layer_norm: bool = True
    final_fc_init_scale: float = 1e-2

    @property
    def actor_hidden_dims(self) -> Tuple[int, int]:
        return (self.hidden_dims, self.hidden_dims)
    
    @property 
    def critic_hidden_dims(self) -> Tuple[int, int]:
        return (self.hidden_dims, self.hidden_dims)

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    momentum: float = 0.9
    b2: float = 0.999
    clip_grad_norm: float = 0.5
    optimizer: str = 'adam'
@dataclass
class PPOConfig:
    """PPO-specific configuration."""
    clipping_ratio: float = 0.25
    gae_lambda: float = 0.5
    num_actor_updates: int = 1
    entropy_coeff: float = 1.0
    temperature: float = 1.0
    store_grads: bool = False
    target_entropy: Optional[float] = None
    spo_loss: bool = False

@dataclass
class TrainingConfig:
    """Training configuration."""
    seed: int = 42
    discount: float = 0.99
    num_critics: int = 5
    
    # Training flags

    min_target: bool = False
    
    # Action space handling
    tanh_squash_distribution: bool = False
    tanh_squash_actions: bool = True




@dataclass
class SuperPPOConfig:
    """Complete configuration for SuperPPO."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_args(cls, args) -> 'SuperPPOConfig':
        """Create config from argparse args."""
        return cls(
            network=NetworkConfig(
                hidden_dims=args.hidden_dims,
                activation_fn=args.activation_fn,
                use_layer_norm=args.use_layer_norm,
            ),
            optimizer=OptimizerConfig(optimizer=args.optimizer),
            ppo=PPOConfig(
                clipping_ratio=args.clipping_ratio,
                gae_lambda=args.gae_lambda,
                entropy_coeff=args.entropy_coeff,
                temperature=args.temperature,
                spo_loss=args.spo_loss,
            ),
            training=TrainingConfig(
                seed=args.seed,
                discount=args.gamma,
                num_critics=args.num_critics,
                min_target=args.min_target,
                tanh_squash_distribution=not args.stable_scheme and args.bound_actions,
                tanh_squash_actions=args.stable_scheme and args.bound_actions
            )
        )




def get_batch(i,batches):
    return  jax.tree.map(lambda x: x[i], batches)


def body(i,val):
    agent,batches = val
    return (agent.update_critics(get_batch(i,batches)),batches)


def scan_body_actor(carry, batch):
    agent, _ = carry  # ignore previous info in carry
    agent, info = agent.update_actor(batch)
    return (agent, info), info  # (new_carry, output)




class Temperature(nn.Module):
    initial_temperature: float = 1.
  
    @nn.compact
    def __call__(self) -> jnp.ndarray:

        log_temp = self.param('log_temp',
                    init_fn=lambda key: jnp.full(
                        (), jnp.log(self.initial_temperature) if self.initial_temperature != 0 else -jnp.inf))
        return jnp.exp(log_temp)

class SACAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    actor: TrainState
    old_actor_params: jnp.ndarray
    temp: TrainState
    old_temp_params : jnp.ndarray
    config: dict = flax_field(pytree_node=False)
    

    #@jax.jit
    def update_critics(agent,batch: Batch):
        
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        def update_one_critic(critic):
                            
                def critic_loss_fn(critic_params):
                        
                        next_actions,next_log_probs,_ = agent.sample_actions(batch["next_observations"],seed=next_key)
                        
                        next_q  = agent.critic.apply_fn({'params': critic_params}, batch['next_observations'], next_actions)
                        
                        target_q = batch['rewards'] + agent.config.training.discount * batch['masks'] * next_q
                        ### Add entropy
                        target_q = target_q - agent.config.training.discount * batch['masks'] * next_log_probs * agent.temp.apply_fn({'params': agent.temp.params})
                        target_q = jax.lax.stop_gradient(target_q)
                        
                        if agent.config.training.min_target:
                            target_q = jnp.min(target_q,axis=0) 
                            target_q = jnp.repeat(target_q.reshape(1,-1),2,axis=0) ## make sure to keep same shape
                           
                        q = agent.critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'])
                        critic_loss = ((q-target_q)**2).mean()
                        
                        return critic_loss, {
                        'critic_loss': critic_loss,
                        'q1': q.mean(),
                    }  
                
                grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
                new_critic = critic.apply_gradients(grads=grads)
                
                return new_critic,critic_info


        new_critics,critic_info = update_one_critic(agent.critic)
        agent = agent.replace(rng=new_rng,critic=new_critics)
        
        return agent

    #@partial(jax.jit,static_argnames=("num_updates",))
    
    @jax.jit
    def update_critics_seq(agent,transitions,num_updates=0 ):
                
        n_batches = transitions['observations'].shape[0]//250

        indexes = jnp.arange(transitions['observations'].shape[0])
        indexes = jax.random.permutation(agent.rng, indexes)
        batch_size = indexes.shape[0] // n_batches
        idxs = indexes[:batch_size * n_batches].reshape((n_batches, batch_size))

        ### Make sure we're doing at least 100 updates per epoch
        ### This is to maintain some fairness between the off-policy and on-policy critics

        #if n_batches <100 or num_updates is not None:

        
        idxs = jax.random.choice(agent.rng, a=transitions['observations'].shape[0], shape=(n_batches, 250), replace=True)

        batches = jax.vmap(lambda i: jax.tree.map(lambda x: x[i], transitions))(idxs)
        agent,batches = jax.lax.fori_loop(0,n_batches,body,(agent,batches))
        
        return agent
    
    @jax.jit
    def update_actor_seq(agent,transitions,num_updates=0 ):
        
        #idxs = jax.random.choice(agent.rng, a=transitions['observations'].shape[0], shape=(num_updates, 250), replace=True)
        n_batches = transitions['observations'].shape[0]//250
        idxs = jnp.arange(transitions['observations'].shape[0])
        idxs = jax.random.permutation(agent.rng, idxs)
        batch_size = idxs.shape[0] // n_batches
        idxs = idxs[:batch_size * n_batches].reshape((n_batches, batch_size))
        
        batches = jax.vmap(lambda i: jax.tree.map(lambda x: x[i], transitions))(idxs)
        
        # Use lax.scan for efficient sequential updates
        # Initialize with proper info structure to match update_actor output
        dummy_info = {
            'actor_loss': 0.0,
            'entropy': 0.0, 
            'approx_kl': 0.0,
            'temp_loss': 0.0,
            'temperature': 0.0,
            'max_ratio':0.0,
            'min_ratio':0.0,
            'percent_outliers':0.0,
            
        }
        initial_carry = (agent, dummy_info)  # (agent, dummy_info)
        (final_agent, _), all_infos = jax.lax.scan(
            scan_body_actor,
            initial_carry,
            batches,
        )
        
        # aggregate across the time dimension (axis=0)
        agg_info = jax.tree_util.tree_map(lambda x: x.mean(0), all_infos)
        return final_agent, agg_info


    
    @partial(jax.jit,static_argnames=("num_updates",))
    def update_critics_seq2(agent,transitions,num_updates=2000 ):
                
        idxs = jax.random.choice(agent.rng, a=transitions['observations'].shape[0], shape=(num_updates, 250), replace=True)

        batches = jax.vmap(lambda i: jax.tree.map(lambda x: x[i], transitions))(idxs)
        agent,batches = jax.lax.fori_loop(0,num_updates,body,(agent,batches))
        
        return agent

    
    @jax.jit
    def update_actor(agent, batch: Batch):
    
    
      
  
        def actor_loss_fn(
                actor_params,
                adv,
                batch,
                #idx,
        ):
        
        
            pre_actions = batch["pre_actions"]
            
            ############### METHOD 2 ###############
        
            dist = agent.actor.apply_fn({'params': actor_params}, batch["observations"])
            new_pre_log_probs = dist.log_prob(pre_actions)
            new_logp = new_pre_log_probs - jnp.sum(2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)), axis=-1)
            
     
            
            dist = agent.actor.apply_fn({'params': agent.old_actor_params}, batch["observations"])            
            old_pre_log_probs = dist.log_prob(pre_actions)
            logp = old_pre_log_probs - jnp.sum(2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)), axis=-1)            
            logratio = new_pre_log_probs - old_pre_log_probs
            
            
            
            
            
            
            #logratio = jnp.clip(logratio, jnp.log(1e-3), jnp.log(1e3))
            
            
            
            ############### METHOD 1 ###############
            # dist = agent.actor(batch["observations"],params=actor_params)
            # pre_actions = batch["pre_actions"]
            # pre_log_probs = dist.log_prob(pre_actions)
            
            # if agent.config["tanh_squash_actions"]:
            #     new_logp = pre_log_probs - jnp.sum(2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)), axis=-1)
            
            # else : 
            #     new_logp = pre_log_probs
            
            
            # logratio = new_logp - batch["log_probs"]
            #######################################
            
            ratio = jnp.exp(logratio)
            
            
         
         
            # Calculate how much policy is changing
            approx_kl = ((ratio - 1) - logratio).mean()

            # Policy loss
            clip_coef = agent.config.ppo.clipping_ratio ##default 0.2 
            masks = batch["masks"]
            outliers = (ratio > 1 + 2 * clip_coef) | (ratio < 1 - 2 * clip_coef)
            
            

            
            
            if agent.config.ppo.spo_loss:
                
                actor_loss_spo_terms = (1.-outliers)* batch["masks"] * adv * ratio - (jnp.abs(batch["masks"] * adv) / (2 * agent.config.ppo.clipping_ratio)) * (ratio - 1)**2
                actor_loss = -actor_loss_spo_terms.mean()
            
            else :
                ### Use standard PPO loss            
                actor_loss1 = masks*adv * ratio
                actor_loss2 = masks*adv * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
                actor_loss = -jnp.minimum(actor_loss1,actor_loss2).mean()
                    
            ### Pad Q and logits because actor buffer is padded ###
            logp = masks * new_logp
            
            entropy = -1 * (masks*logp).sum()/(masks.sum())
            
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': entropy,
                'approx_kl':approx_kl,
                'max_ratio':jnp.max(ratio),
                'min_ratio':jnp.min(ratio),
                'percent_outliers': jnp.mean((ratio > 1 + 2 * clip_coef) | (ratio < 1 - 2 * clip_coef)),
              
            }
            
        
        def temp_loss_fn(temp_params, entropy, target_entropy):
            """
            Calculates temperature loss but forces the loss to 0 if the new
            temperature is outside the allowed ratio of the old temperature.
            """
            new_temperature = agent.temp.apply_fn({'params': temp_params})
            old_temperature = agent.temp.apply_fn({'params': agent.old_temp_params})
            clip_coef = agent.config.ppo.clipping_ratio
            
            # Ensure old_temperature is treated as a non-differentiable constant
            old_temperature = jax.lax.stop_gradient(old_temperature)
            
            # --- Hard Clipping Logic ---
            
            # 1. Calculate the ratio
            temp_ratio = new_temperature / (old_temperature + 1e-8)
            
            # 2. Define the allowed range and the condition
            min_ratio = 1.0 - clip_coef
            max_ratio = 1.0 + clip_coef
            is_within_bounds = (temp_ratio >= min_ratio) & (temp_ratio <= max_ratio)
            
            # 3. Calculate the standard loss
            # We stop the gradient on the error term as is standard practice.
            entropy_error = jax.lax.stop_gradient(entropy - target_entropy)
            standard_loss = new_temperature * entropy_error
            
            # 4. Use jnp.where to enforce the hard clip.
            # If `is_within_bounds` is True, use `standard_loss`.
            # If `is_within_bounds` is False, use `0.0`.
            temp_loss = jnp.where(is_within_bounds, standard_loss, 0.0)
            temp_loss = jnp.where((new_temperature < 0.01) & (temp_loss > 0), 0.0, temp_loss.mean())
            
       
            
            return temp_loss.mean(), {
                'temp_loss': temp_loss,
                'temperature': new_temperature,
            }
            
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        observations,next_observations = batch["observations"],batch["next_observations"]
        
        observations = jnp.concatenate([observations, next_observations[-1][None]], axis=0)
        
        def evaluate(observations,key):
            
            actions, log_p,_ = agent.sample_actions(observations,seed=key)
            q_all = agent.critic.apply_fn({'params': agent.critic.params}, observations, actions)
            v = jnp.mean(q_all,axis=0)
            
            return v,log_p
        
   

        ### Compute advantage for the fixed states AND actions
        vs,hs = jax.vmap(evaluate,in_axes=(None,0))(batch["observations"],jax.random.split(curr_key,10))        
        tmp_v,tmp_logp = jnp.mean(vs,axis=0),jnp.mean(hs,axis=0)
        q = agent.critic.apply_fn({'params': agent.critic.params}, batch["observations"], batch["actions"]).mean(axis=0)
        
        
        dist = agent.actor.apply_fn({'params': agent.old_actor_params}, batch["observations"])
        pre_actions = batch["pre_actions"]
        pre_log_probs = dist.log_prob(pre_actions)
        
        if agent.config.training.tanh_squash_actions:
            logp = pre_log_probs - jnp.sum(2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)), axis=-1)
        
        logp = jax.lax.stop_gradient(logp)
        
        
        adv = (q-agent.temp.apply_fn({'params': agent.temp.params})*logp) - (tmp_v - agent.temp.apply_fn({'params': agent.temp.params}) *tmp_logp)### This one worked
        adv = adv.reshape(-1)
        # Normalize advantages
        #adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
        

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params, adv, batch)
        new_actor = agent.actor.apply_gradients(grads=grads)
        grads, temp_info = jax.grad(temp_loss_fn, has_aux=True)(agent.temp.params, actor_info['entropy'], agent.config.ppo.target_entropy)
        new_temp = agent.temp.apply_gradients(grads=grads)
        #new_temp,temp_info = agent.temp,{"temp_loss":0.0,"temperature":agent.temp()}
        
        agent = agent.replace(rng=new_rng, actor=new_actor,temp=new_temp)

        info = {**actor_info, **temp_info}  
    
        return agent,info
                    
                    

            
        
    @jax.jit
    def sample_actions(agent,   
                       observations: np.ndarray,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       params=None,
                       ) -> jnp.ndarray:
        
        ### random always true
        if params is None:
            params = agent.actor.params
        dist = agent.actor.apply_fn({'params': params}, observations, temperature=temperature)
        pre_actions,pre_log_ps = dist.sample_and_log_prob(seed=seed)
        
        if agent.config.training.tanh_squash_actions:
            actions = jax.nn.tanh(pre_actions)
            log_ps = pre_log_ps - jnp.sum(2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)), axis=-1)        
        
        else : 
            log_ps = pre_log_ps
            actions = pre_actions
            
        return actions,log_ps,pre_actions
    
    
    @jax.jit
    def deterministic_action(agent,   
                       observations: np.ndarray,
                       ) -> jnp.ndarray:
        
        ### random always true
        seed = jax.random.PRNGKey(0)
        dist = agent.actor.apply_fn({'params': agent.actor.params}, observations, temperature=0.)
        pre_actions,pre_log_ps = dist.sample_and_log_prob(seed=seed)
        if agent.config.training.tanh_squash_actions:
            actions = jax.nn.tanh(pre_actions)
        
        else :
            actions = pre_actions
        
        return actions



def create_learner(
    config: SuperPPOConfig,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
):

    """
    Create a PPO learner using structured configuration.
    
    Args:
        config: SuperPPOConfig object containing all configuration
        observations: Example observations for network initialization
        actions: Example actions for network initialization
    """
    
    rng = jax.random.PRNGKey(config.training.seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    activations = nn.relu if config.network.activation_fn == 'relu' else nn.tanh
    final_fc_init_scale = config.network.final_fc_init_scale

    action_dim = actions.shape[-1]
    actor_def = Policy(
        config.network.actor_hidden_dims, 
        action_dim=action_dim,
        activations=activations,
        final_fc_init_scale=final_fc_init_scale,
        tanh_squash_distribution=config.training.tanh_squash_distribution,
        use_layer_norm=config.network.use_layer_norm,
    )

    critic_def = ensemblize(OriginalCritic, config.training.num_critics)(
        hidden_dims=config.network.critic_hidden_dims,
        use_layer_norm=config.network.use_layer_norm,
        activations=activations
    )
    critic_params = critic_def.init(critic_key, observations, actions)['params']
    critic = TrainState.create(
        apply_fn=critic_def.apply, 
        params=critic_params, 
        tx=optax.adam(learning_rate=config.optimizer.critic_lr)
    )
      
   
    actor_params = actor_def.init(actor_key, observations)['params']
    temp_def = Temperature(config.ppo.temperature)
    temp_params = temp_def.init(rng)['params']
    
    
    
    
    if config.optimizer.optimizer == "adamw":
        tx = optax.adamw(
            learning_rate=config.optimizer.actor_lr,
            b1=config.optimizer.momentum,
            b2=config.optimizer.b2
        )
    elif config.optimizer.optimizer == "adam":
        tx = optax.adam(
            learning_rate=config.optimizer.actor_lr,
            b1=config.optimizer.momentum,
            b2=config.optimizer.b2
        )
    elif config.optimizer.optimizer == "rmsprop":
        tx = optax.rmsprop(
            learning_rate=config.optimizer.actor_lr,
            momentum=config.optimizer.momentum,
            b2=config.optimizer.b2
        )
    elif config.optimizer.optimizer == "sgd":
        tx = optax.sgd(
            learning_rate=config.optimizer.actor_lr,
            momentum=config.optimizer.momentum
        )
        
    tx = optax.chain(
        optax.clip_by_global_norm(config.optimizer.clip_grad_norm),
        tx
    )
    

    actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=tx)
    temp = TrainState.create(
        apply_fn=temp_def.apply, 
        params=temp_params, 
        tx=optax.adam(
            learning_rate=config.optimizer.temp_lr,
            b1=config.optimizer.momentum,
            b2=config.optimizer.b2
        )
    )
        
    target_entropy = config.ppo.target_entropy
    if target_entropy is None:
        target_entropy = -config.ppo.entropy_coeff * action_dim

    if config.ppo.target_entropy is None:
        config.ppo.target_entropy = -config.ppo.entropy_coeff * action_dim


    return SACAgent(
        rng,
        critic=critic,
        actor=actor,
        old_actor_params=actor_params,
        temp=temp,
        old_temp_params=temp_params,
        config=config,
    )




