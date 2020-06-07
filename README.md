#### [Code Repo](https://github.com/nfragakis/nfragakis.github.io)
#### [Content Theory Doc](https://docs.google.com/document/d/1zh4VayKWxYLirSwCZilDiCEzKVXImzz4uGE0EDfc9_A/edit#)
#### [Implementation Walkthrough](https://youtu.be/n5nuqHigvT0)
#### [Train you own Agent](https://colab.research.google.com/drive/1gCgDN338dJ9RNEUIW0nSg8B2nxEWRn4c?usp=sharing)

<a href="https://youtu.be/EJ3LfvFKxgs " target="_blank"><img src="https://img.youtube.com/vi/EJ3LfvFKxgs/0.jpg" 
alt="Simulation Video" width="360" height="240" border="0" /></a>

## Description of the Domain
In this project, I propose a solution for formulating the emotions of virtual agents as they interact with a 
simulated world, learning how to walk and interact with their environment. We will establish a basic understanding 
of the Reinforcement Learning paradigm, see how this compares to biological entities, and establish the context and 
details from which we will formulate emotions, representing the internal state of our agents as they interact 
with and learn from their environment.

## Why we care about this domain
Reinforcement Learning (RL) loosely models the way humans learn. As an agent 
interacts with the environment it earns a reward proportional to its performance;
similar to Dopamine, the neurotransmitter which helps to moderate human emotion. 
This represents a completely different paradigm than modern supervised learning 
techniques, which learn entirely from historical examples in the training data. 
Because our RL agent’s learning is not tied to the past, it can arrive at completely
novel and creative solutions to the problems it faces. Through developing a model of it's
environment, the agent plans ahead, predicting outcomes for a range of possible actions
and choosing the one it thinks will lead to the best outcome. By developing a 
content theory of the emotions present in these agents as they learn, I hope to gain
a better understanding of how humans learn in the real world, as well as improving the
training and performance of these virtual agents. In developing AI that no longer learns
through historical data, but by it’s own interactions with an environment, I believe it
is only a matter of time until we see an intelligence explosion that is completely seperate from that of human
from our own intelligence. 

## Defined Terms in this Content 
- **Agent** - Makes decisions in a simulation based on a policy learned through interactions with an environment which provides reward and punishment to guide effective behaviour.
- **Environment**- Simulated world the agent lives in and interacts with.
- **State** - Complete description of the environment (Position of agents limbs, speed, lidar rangefinder, and angular measurements) 
- **Action Space** - Set of all valid actions an agent can take within an environment. Could be continuous, such as controlling a simulated robot, or discrete, such as playing an Atari game.
- **Reward**- Signal that an agent perceives from an environment signifying the quality of a given state. The agent’s goal is to learn the policy/behaviour that maximizes this value.
- **Value Function** - Mapping a particular state-action pair to the agents expected reward of this action. Primary learning objective in RL.
- **Policy** - A rule or decision framework that an agent uses to interact with its environment can be deterministic or stochastic in nature. (We generally prefer stochastic in training so as to encourage novel behaviours and experimentation within an environment).
- **Somatic Marker** - feelings in the body that are associated with emotions. Somatic Markers strongly influence decision-making at the unconscious level.
- **Joy** - Experienced when the agents reward from the environment and positive expectations of that reward overshadow the previous moments valuation.
- **Distress** - Feeling the agent gets when it’s expectations and reality drop below its previous valuation of the environment.
- **Hope** - The agent carries positive expectations of future events.
- **Fear** - Negative expectations within the agent of how future events will play out.
- **Uncertainty** - The clash between an agent's expectation of how future events will play out with the ground reality of the situation. 

## Reinforcement Learning 
- Reinforcement Learning (RL), a sub-field of AI, consists of an agent who learns a policy, or set of behaviors, over time through episodic interactions with its environment.
- The agent’s actions are chosen based on its inputs, representing the current state of the environment, which can be likened to sensory inputs of biological entities.
- In order to identify the best action in a situation, our agent develops a policy that maps a given state and action to the expected reward, very similar
    to how this mechanism functions in biological entities.
- Over time these mappings of state to action become habitual leading to consistent behaviour emerging.

## Somatic Marker Hypothesis
- Somatic Markers are feelings within one's body, that over time become associated with certain emotional states. 
- Markers have a profound impact on decision-making in biological agents, particularly regarding the ability to make fast decisions in complex environments. 
- These Somatic Markers and the actions that arise from them are the foundation of "Gut-Feelings" or Intuition that we unconsciously experience.
- Over time these bodily states and subsequent emotions become linked to particular situations faced and the resulting outcome, similar to the RL framework outlined above.
- These structures in cognition are absolutely essential for humans and other biological entities to be able to react rapidly in situations without having to employ rationality or logic.

## Formulation of Emotions
- At a high-level, we use a Neural Network model to act as the primary pattern matching function, taking in the state variables
from the environment and oututting a specific action to maximize the expected return to our agent.
- By developing formulas representing an agent's internal state as it is interacting with the environment, we are able to 
gain insight into how the learning process is unfolding.


#### Joy & Distress
- Joy and Distress are experienced as the outcome of the situation, plus our expectations of that outcome 
are balanced with the previous emotional state we are coming from.
- In RL Terms this is represented as the reward received from the interaction with the environment plus the
expected reward, minus the reward received at the previous timestep. When this is postive we experience Joy,
when negative this is formulated as Distress.

<img src="https://latex.codecogs.com/gif.latex?Joy/Distress = R(s_t,a_t) + Q^{\pi}(s,a) - R(s_{t-1}, a_{t-1})">

#### Hope & Fear 
- Hope and Fear are a direct measure of our expectations about the future, within the context of the actions 
we plan on taking.
- In RL terms this is fairly simply to formulate as our value function does exactly this, matching an state action
pair to our expected reward or outcome. When this is postive the agent experiences Hope and Fear when negative.

<img src="https://latex.codecogs.com/gif.latex?Hope/Fear = Q^{\pi}(s,a)">

#### **Unertainty**
- Uncertainty is defined as the class between our expectations of reality, and the underlying truth. What actually occurs.
- In RL terms, we use the differentiable optimization function for our Q-Value Neural Network. The Mean-Squared Error of 
our expected outcome, or value of the chosen action within our current state, against the reward actually received by the 
environment.

<img src="https://latex.codecogs.com/gif.latex?Uncertainty = \frac{1}{n}\sum_{i=1}^{n}(Q^{\pi}(s,a) - R(s_t, a_t))^2">

## Content Theory Conclusions
**Conclusions** - By modeling the structures from which humans learn in a virtual setting,
we open up the possibility to understand the internal state of an agent at all times
throughout this process, leading to an improvement in the understanding of how both
artificial and biological agents learn through interactions with their environment.

**Summary** - In this paper, we have detailed the framework of Reinforcement Learning, 
how a software agent in a virtual simulation is able to learn optimal behavior through episodic 
interactions with its environment. We discussed the many parallels between this process and the 
Somatic Marker Hypothesis, which details how biological entities map specific sensory inputs from 
environment to actions, emotions, and outcomes. Finally, we outline the process from which we can 
formulate the emotions of our artificial agents as they learn, leading to a better understanding 
of the internal processes and emotional states that arise in both artificial and biological agents.


## Walkthrough of Implementation

### Simulation and Scope
<a href="https://youtu.be/EJ3LfvFKxgs " target="_blank"><img src="https://img.youtube.com/vi/EJ3LfvFKxgs/0.jpg" 
alt="Simulation Video" width="320" height="240" border="0" /></a>

The initial simulation takes place in the OpenAI Gym Python library's [Bipedal-Walker Environment](https://gym.openai.com/envs/BipedalWalker-v2/).
The agent's goal is to develop an optimal policy to get from one end of the 2-d simulated environment to the other. The agent receives a reward 
of 300 points if it makes it to the end and -100 points if it falls. Additionally, a small reward is received for each forward step it takes.
Any form of torque applied comes with a small penalty, thus the agent is guided to complete it's task as efficiently as possible. The state values 
that the agent has access to include it's Speed, Position of Joints, Lidar Rangefinder Measurements, and other Angular characteristics.

### Architecture
#### Model 
The primary model used in my implementation is the Deep Deterministic Policy Gradient (DDPG) algorithm, or Actor-Critic model
as it is sometimes entitled. This model uses Neural Networks as the primary pattern matching function to map a state-action 
pair to an incentive value, and ultimately an action within our simulation. This model concurrently learns a Q-Function, which 
maps each state-action pair to it’s value, and a policy, which learns optimal actions through interpreting the Q-Functions mappings.
Because our model learns both of these functions in parallel, it is able to incorporate exploration into its actions, 
as opposed to acting in a deterministic manner based on the highest value in our Q-Function. 
The policy allows room for random exploration and avoids getting stuck in a local optimum.

For more information check out the documentation [here](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

```python

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            action = self.pi(obs).numpy() 

            # Random Noise inserted into Agents Action
            action += noise * np.random.randn(act_dim)
            return np.clip(action, -act_limit, act_limit)
```

As we have two seperate Neural Networks present in our model, we follow two seperate steps in order to calculate our loss functions,
the first, for the Q Function, involves taking in a batch of environment interactions from our replay buffer, computing the 
state-action pair values for each action within the domain, and comparing this to target network as discussed [here](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#the-q-learning-side-of-ddpg)

```python

def compute_loss_q(data, net, targ_net, gamma):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

    q = net.q(o,a)

    # Bellman backup for Q function
    with torch.no_grad():
        q_pi_targ = targ_net.q(o2, targ_net.pi(o2))
        backup = r + gamma * (1 - d) * q_pi_targ

    # MSE loss against Bellman backup
    loss_q = ((q - backup)**2).mean()

    # Useful info for logging
    loss_info = dict(QVals=q.detach().numpy())

    return loss_q, loss_info
```

The loss function for our Policy Net is much simpler. We simply take observations from our recorded interactions with the 
environment and allow the Q-Function to calculate an expected value based on the observation and return the negative of the output (must use negative to properly adjust Neural Net paramters)
For more information about this check out this [link](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#the-policy-learning-side-of-ddpg)

```python

def compute_loss_pi(data, net):
    o = data['obs']
    q_pi = net.q(o, net.pi(o))
    return -q_pi.mean()

```

The actual parameter update function is out of the scope of this overview, but can be found in the [run.py](https://github.com/nfragakis/Formulating-Emotions-in-Reinforcement-Learning-Agents/blob/master/run.py) 
file of the repository.


##### Replay Buffer
Instead of learning at each step of the simulation, which would not enable us to act in a timely manner within our 
simulation, the model makes use of a Replay Buffer. This buffer stores the relevant information such as the state 
of the environment, action taken, reward, and subsequent state the agent finds itself in. Once the agent has enough 
of these observations stored up, it then runs its update step, adjusting the parameters of our Actor-Critic Model 
through backpropagation. For more information on this algorithm check out this link 

```python

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
```

The implementation for the model and replay buffer can be found in the [core.py](https://github.com/nfragakis/Formulating-Emotions-in-Reinforcement-Learning-Agents/blob/master/core.py) file available in the code repository
#### Running Experiments 
The main experimentation code for my implementation lies in the run.py file.  It was designed to be run from the command
line and has a number of different parameters available to tune experimentation such as batch size for our neural nets, 
number of training epochs, naming of the experiment directory, etc. 

```console
!python run.py --exp_name Emotions --epochs 3000 --env 'BipedalWalker-v3' --batch_size 128
```

A number of other helper files are present in the lib folder and contain functions to support increased efficiency 
in the update processes for our model, logging functionality of our formulated emotions, and video monitoring of our agent 
as it learns.

A typical interaction with our environment consists of choosing an action based on our observation of the state,
stepping the environment forward, and storing information in our replay buffer as seen below.

```python
o, ep_ret, ep_len, r = env.reset(), 0, 0, 0

a = get_action(o)

# Next observation, reward, done?
o2, r, d, _ = env.step(a)

replay_buffer.store(o, a, r, o2, d)
```


For a detailed walkthrough of the code and examples on how to run the experiments check out my [implementation video](https://youtu.be/n5nuqHigvT0) and run your own experiments
in the google Colab environment [Colab Notebook](https://colab.research.google.com/drive/1gCgDN338dJ9RNEUIW0nSg8B2nxEWRn4c?usp=sharing).

### Results
After running a number of experiments and averaging out the results, a couple of patterns emerge. First, with regard 
to Joy and Distress, the agent generally starts out with a large degree of naive overconfidence. It is unfamiliar with 
the environment and thus does not know what to expect. Because of this there is a large amount of Joy, but also, an above 
average amount of distress as well. This is primarily due to the unstable nature of the emotions at this stage and the high
variability present in our models expectations of rewards. As the training begins to stabilize, we see the joy value drop 
sharply and remain low until the agent finally begins to understand and operate efficiently within the environment.
Distress; however, remains fairly high unil the agent begins to learn an optimal policy.  The same trend holds for hope and
fear. The agent initially starts out very hopeful about the environment, meaning it is expecting a high reward for its actions.
This quickly plummets as it gains more experience, before ultimately shooting back up once it gains an understanding of what to
expect. The results for uncertainty are interesting, in that they follow a similar pattern of starting off very high, tailing
off, and then rising towards the end of training. Initially, this makes sense as we would expect to have a high degree of 
uncertainty, and subsequently updates to our model, early on. However, the high spike around epoch 400 is a bit surprising. 
For future experiments I am interested in running the training process for longer and in more complex environments; however, 
due to limits in time and computational power, we are left with the below results, transpiring from around 12 hours of training
on a GPU.

#### Joy vs Distress
![Joy vs. Distress](https://github.com/nfragakis/Formulating-Emotions-in-Reinforcement-Learning-Agents/blob/master/data/Joy:Distress.png?raw=true)

#### Hope vs. Fear
![Hope vs. Fear](https://github.com/nfragakis/Formulating-Emotions-in-Reinforcement-Learning-Agents/blob/master/data/Hope:Fear.png?raw=true)

#### Uncertainty 
![Uncertainty](https://github.com/nfragakis/Formulating-Emotions-in-Reinforcement-Learning-Agents/blob/master/data/Uncertainty.png?raw=true)

#### Reward
![Reward](https://github.com/nfragakis/Formulating-Emotions-in-Reinforcement-Learning-Agents/blob/master/data/Reward.png?raw=true)

## Suggested Readings
- **Spinning Up in Deep RL (Open AI)**
    - [Website](https://spinningup.openai.com/en/latest/index.html)
    - Best, code first, introduction to Deep Reinforcement Learning I was able to find, highly recommend
- **Reinforcement Learning in artificial and biological systems**
    - Neftci, E.O., Averbeck, B.B. Reinforcement learning in artificial and biological systems. Nat Mach Intell 1, 133–143 (2019).
    - [Paper](https://www.nature.com/articles/s42256-019-0025-4?draft=marketing)
    - Great intro linking state of the art research in both biology and computer science
- **AlphaGo Documentary**
    - Great Documentary about Deep Minds RL Agent that took on the worlds greatest professional Go Player
    - Available for free on youtube [video](https://www.youtube.com/watch?v=WXuK6gekU1Y)
- **Introduction to Reinforcement Learning**
    - Richard S. Sutton and Andrew G. Barto. 1998. Introduction to Reinforcement Learning (1st. ed.). MIT Press, Cambridge, MA, USA.
    - [Book Download (distributed freely)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
    - Universally recognized as the most comprehensive introductory textbook on the subject of RL.

- **David Siver (Deep Mind) Intro to RL Course**
    - [Course Link](https://www.davidsilver.uk/teaching/)
    - One of the leaders of the RL team at DeepMind, famous for many breaking achievements such as defeating the World Champion at Go and Starcraft 

- **OpenAI multi-agent hide and seek**
    - [Blog Post](https://openai.com/blog/emergent-tool-use/)
    - Breakthrough work in RL where agents learn to cooperate and develop highly creative strategies in order to beat another team of agents in Hide and Go Seek. Highly recommend checking this work out, amazing what they were able to accomplish.








    



