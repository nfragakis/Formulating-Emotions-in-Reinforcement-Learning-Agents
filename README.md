#### [Code Repo](https://github.com/nfragakis/nfragakis.github.io)
#### [Content Theory Doc](https://docs.google.com/document/d/1zh4VayKWxYLirSwCZilDiCEzKVXImzz4uGE0EDfc9_A/edit#)
#### [Ideas File](https://docs.google.com/document/d/1c3yM_woKnLbukBI9sGDSJJ4zpply-S0q2qNzZN6L4VE/edit)
#### [Markdown Cheat-Sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)


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

## Defined Terms in this content 
- **Agent** - Makes decisions in a simulation based on a policy learned through interactions with an environment which provides reward and punishment to guide effective behaviour.
- **Environment**- Simulated world the agent lives in and interacts with.
- **Reward**- Signal that an agent perceives from an environment signifying the quality of a given state. The agent’s goal is to learn the policy/behaviour that maximizes this value.
- **State** - Complete description of the environment (Position of agents limbs, speed, lidar rangefinder, and angular measurements) 
- **Action Space** - Set of all valid actions an agent can take within an environment. Could be continuous, such as controlling a simulated robot, or discrete, such as playing an Atari game.
- **Policy** - A rule or decision framework that an agent uses to interact with its environment can be deterministic or stochastic in nature. (We generally prefer stochastic in training so as to encourage novel behaviours and experimentation within an environment).
- **Memory Buffer** - Queue of past simulation events containing observations on status of environment, actions taken, and the overall reward and outcome derived from action.
- **Policy Gradient** - technique that relies on optimizing parameterized policies w.r.t the expected long-term cumulative return by gradient descent (Neural Networks).
- **Value Function** - The agent’s expected return or reward from the environment if it follows out its policy.
- **Bellman Equations** - A set of dynamic programming algorithms critical in Reinforcement Learning, in which the value of your starting point is your return of the current state plus the expected value of future states when following out a policy.
- **Trajectories** - Sequence of states and actions (often called episodes).

## Objects in the Domain
- **Simulation Environment** - 2D BipedalWalker simulation in which the agent must learn how to walk from one end of environment to another as efficiently as possible.
- **Agent** 
- **State** 
- **

## Reinforcement Learning 
- Reinforcement Learning (RL), a sub-field of AI, consists of an agent who learns a policy, or set of behaviors, over time through episodic interactions with its environment.
- The agent’s actions are chosen based on its inputs, representing the current state of the environment, which can be likened to sensory inputs of biological entities.
- Punishments and rewards returned to the agent are similar to the biological dopamine mechanisms which help to regulate human behavior.
- Agent maintains a bank of memories from which it draws from in the learning process.

## Somatic Marker Hypothesis
- Somatic Markers are feelings within one's body, that over time become associated with certain emotional states. 
- Markers have a profound impact on decision-making in biological agents, particularly regarding the ability to make fast decisions in complex environments. 
- Over time these bodily states and subsequent emotions become linked to particular situations faced and the resulting outcome, similar to the RL framework outlined above.
- Through repetition these marker-action pairs become further fortified, leading to consistent behaviors being adapted by a person.
- Consider the example of walking in the woods and seeing a snake.
- These structures in cognition are absolutely essential for humans and other biological entities to be able to react rapidly in situations without having to employ rationality or logic.


## Formulation of Emotions
##### **Joy**
##### **Distress**
##### **Hope**
##### **Fear**
##### **Certainty**
##### **Satisfaction**
##### **Frustration**

## Content Theory Conclusions

## Walkthrough of Implementation

### Simulation and Scope
<a href="https://youtu.be/EJ3LfvFKxgs " target="_blank"><img src="https://img.youtube.com/vi/EJ3LfvFKxgs/0.jpg" 
alt="Simulation Video" width="320" height="240" border="0" /></a>

<u>Click image to view video</u>

The initial simulation takes place in the OpenAI Gym Python library's [Bipedal-Walker Environment](https://gym.openai.com/envs/BipedalWalker-v2/).
The agent's goal is to develop an optimal policy to get from one end of the 2-d simulated environment to the other. The agent receives a reward 
of 300 points if it makes it to the end and -100 points if it falls. Additionally, a small reward is received for each forward step it takes.
Any form of torque applied comes with a small penalty, thus the agent is guided to complete it's task as efficiently as possible. The state values 
that the agent has access to include it's Speed, Position of Joints, Lidar Rangefinder Measurements, and other Angular characteristics.

## Blue Sky Ideas 

## Technical Addendum
#### Bellman Equations 
#### Q-Learning
#### Policy Gradient
#### Actor-Critic Model 
#### Current State of the Art and Future Possibilities in RL

## Suggested Readings

## Bibliography


- Content Theory
    - [Read Somatic Marker Paper](https://www.brainmaster.com/software/pubs/brain/Dunn%20somatic_marker_hypothesis.pdf)
    - [Castlefranchi Felt Emotions (p. 75](https://d2l.depaul.edu/d2l/le/content/745964/viewContent/6387839/View)
    - [Somatic Wiki](https://en.wikipedia.org/wiki/Somatic_marker_hypothesis)
    - [Emotion Driven RL](https://pdfs.semanticscholar.org/0818/f199953a13fd933759beb8b2f461225c1cd8.pdf)
    - [Emotion in Q-Learning](https://arxiv.org/pdf/1609.01468.pdf)
    - [Emotions in RL Agents](https://arxiv.org/pdf/1705.05172.pdf)
    - [Social Influence in Multi-Agent RL](https://arxiv.org/pdf/1810.08647.pdf)
    - [Open AI Learning to Cooperate](https://openai.com/blog/learning-to-cooperate-compete-and-communicate/)
    - [Open AI Learning to Communicate](https://openai.com/blog/learning-to-cooperate-compete-and-communicate/)
    - [Joy, Destress, Hope, Fear in RL](https://dl.acm.org/doi/10.5555/2615731.2616089O)
    - [DDPG Paper](https://arxiv.org/abs/1509.02971)
    - [RL in Bio and Artificial Agents](https://www.nature.com/articles/s42256-019-0025-4)
    - [OCC model of Emotions](https://journals-sagepub-com.ezproxy.depaul.edu/doi/10.1177/1754073913489751)
    - [Blog Posting (Somatic Markers)](https://imotions.com/blog/somatic-marker-hypothesis/)



    



