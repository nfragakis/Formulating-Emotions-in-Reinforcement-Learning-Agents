import argparse
import time
from copy import deepcopy
import gym
import numpy as np
import torch
from torch.optim import Adam
import core
from lib.logx import EpochLogger
from lib.run_utils import setup_logger_kwargs

DEFAULT_ENV_NAME =  'BipedalWalker-v3'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pi_lr', type=float, default=1e-3)
    parser.add_argument('--q_lr', type=float, default=1e-3)
    parser.add_argument('--act_noise', type=float, default=0.1)
    parser.add_argument("--record", "-r", help="Directory to store video recording")
    parser.add_argument('--num_test_episodes', type=int, default=15)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--update_every', type=int, default=50)
    args = parser.parse_args()

    # Build experiment folder structure
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)

    # Make simulation environment
    env_fn = lambda : gym.make(args.env)
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape 
    act_dim = env.action_space.shape[0] 
    act_limit = env.action_space.high[0]

    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    # Instantiate Actor Critic Neural Net and Target Network
    net = core.MLPActorCritic(env.observation_space, env.action_space)
    targ_net = deepcopy(net)
    
    # Freeze target network
    for p in targ_net.parameters():
        p.requires_grad = False
 
    # Experience / Memory Buffer
    replay_buffer = core.ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(1e6))

    # Count number of parameters in network
    param_counts = tuple(core.count_vars(module) for module in [net.pi, net.q])
    logger.log('\nNumber of Parameters: \t pi: %d, \t q: %d\n'%param_counts)
    # Set up optimization functions for policy and q-function
    pi_optimizer = Adam(net.pi.parameters(), lr=args.pi_lr)
    q_optimizer = Adam(net.q.parameters(), lr=args.q_lr)


########################################################################################################################
    """ Currently need functions here as they rely on cmd line args from run file """

    def update(data, net=net, targ_net=targ_net, gamma=args.gamma, polyak=args.polyak, 
               q_optimizer=q_optimizer, pi_optimizer=pi_optimizer):
        """    
        Update process for Actor Critic Neural Network
        """ 
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = core.compute_loss_q(data, net, targ_net, gamma)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in net.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = core.compute_loss_pi(data, net)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in net.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(net.parameters(), net.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


    ### MOVE TO CORE?
    def get_action(o, noise_scale, net=net):
        a = net.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(render=False):
        for j in range(args.num_test_episodes):
            if render:
                test_env.render()
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == args.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0, net))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

########################################################################################################################

    # Setup model saving
    logger.setup_pytorch_saver(net)

    # Prepare for interaction with environment
    total_steps = args.steps_per_epoch * args.epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    

    # MAIN TRAINING LOOP
    for t in range(total_steps):

        # sample random actions until start_steps 
        # when memory buffer is built up 
        if t > args.start_steps:
            a = get_action(o, args.act_noise)
        else:
            a = env.action_space.sample()

        # Step the environment 
        o2, r, d, _ = env.step(a)
        ep_ret += r 
        ep_len += 1 

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==args.max_ep_len else d

        # Store experience in replay buffer 
        replay_buffer.store(o, a, r, o2, d)

        # Update most recent observation of state 
        o = o2 

        # End of trajectory handling 
        if d or (ep_len == args.max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update Neural Net Parameters 
        if (t >= args.update_after) and (t % args.update_every == 0):
            for _ in range(args.update_every):
                # pull random batch of experiences from memory buffer
                batch = replay_buffer.sample_batch(args.batch_size)
                update(batch)

        # End of Epoch handling
        if (t+1) % args.steps_per_epoch == 0:
            epoch = (t+1) // args.steps_per_epoch 

            # Save Model
            if (epoch % args.save_freq == 0) or (epoch == args.epochs):
                logger.save_state({'env':env}, None)

            # Test the performance of the deterministic version of the agent.
            if epoch % 50 == 0:
                test_agent(render=True)
            else:
                test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

