# Formulating Emotions in Reinforcement Learning Agents

## [Ideas File](https://docs.google.com/document/d/1c3yM_woKnLbukBI9sGDSJJ4zpply-S0q2qNzZN6L4VE/edit)
## [Markdown Cheat-Sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

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
- **Agent** - Makes decisions in a simulation based on a policy learned through interactions with an environment which provide rewards and punishment to guide effective behaviour.
- **Environment**- Simulated world the agent lives in and interacts with.
- **Reward**- Signal that an agent perceives from an environment signifying the quality of a given state. The agent’s goal is to learn the policy/behaviour that maximizes this value.
- **State** - Complete description of the state of the environment (RGB matrix of pixel values, position of agents limbs, etc..).
- **Action Space** - Set of all valid actions an agent can take within an environment. Could be continuous, such as controlling a simulated robot, or discrete, such as playing an Atari game.
- **Policy** - A rule or decision framework that an agent uses to interact with its environment can be deterministic or stochastic in nature. (We generally prefer stochastic in training so as to encourage novel behaviours and experimentation within an environment).
- **Policy Gradient** - technique that relies on optimizing parameterized policies w.r.t the expected long-term cumulative return by gradient descent (Neural Networks).
- **Value Function** - The agent’s expected return or reward from the environment if it follows out its policy.
- **Bellman Equations** - A set of dynamic programming algorithms critical in Reinforcement Learning, in which the value of your starting point is your return of the current state plus the expected value of future states when following out a policy.
- **Trajectories** - Sequence of states and actions (often called episodes).

## TODO 
- Add Emotion Functions to Model 
    - Formulate emotions based off core RL features in Algo files
        - reward function (incorporate emotions?)
        - emotion functions (from types above)
- Split code repo and website into seperate repos
- Content Theory / Main webpage README
    - Write Wiki on Reinforcement Learning 
    - Connect concepts to content theory 
    - Explain SIM w/ Video Examples
    - Technical Addendum on RL and mathematics of DDPG algo
- README for running sims (through Colab and locally)


### Emotion Types
- Joy (tied to Reward Function and Expectation (policy)
    - Joy = (Reward from the environment + our expected value of that
    reward — our value of the previous state we were in) * an uncertainty term 
    - Previous state matters, if I'm already feeling good I'm more
    likely to feel good in future time steps.
    - Joy = (R(State,Action) + Q(State,Action) — Q(State, Action for Previous 
    State Action Pair)
- Gut Feeling
    - “Gut Feeling in computational terms is the exploration vs exploitation
    dilemma, most implementations are stochastic based on fixed value. How do
    we add more intuition to agents process? 
    - (average certainty over time %right in Q(s,a)
- Hope
    - Positive Expected Value of Reward 
    - (Must add these formulas in prior to completion of Q(s,a) step)
- Fear
    - Negative Expected Value of Reward
- Global Certainty
    -  level of certainty regarding the outcome of a specific state action pair. 
    It is a function of our historic level of accuracy in predicting the outcome 
    of that state action pair.
    - Average value of historic errors?
- Local Uncertainty
    - (mean of historic errors for specific Q(S,A) + Most recent error 
    for that Q(S,A) ) / 2

- Content Theory
    - [Read Somatic Marker Paper](https://www.brainmaster.com/software/pubs/brain/Dunn%20somatic_marker_hypothesis.pdf)
    - [Castlefranchi Felt Emotions (p. 75](https://d2l.depaul.edu/d2l/le/content/745964/viewContent/6387839/View)
    - [Somatic Wiki](https://en.wikipedia.org/wiki/Somatic_marker_hypothesis)
    - [Emotions in RL Medium](https://medium.com/datadriveninvestor/reinforcement-learning-towards-an-emotion-based-behavior-system-73e833c1ba75)
    - [Emotion Driven RL](https://pdfs.semanticscholar.org/0818/f199953a13fd933759beb8b2f461225c1cd8.pdf)
    - [Emotion in Q-Learning](https://arxiv.org/pdf/1609.01468.pdf)
    - [Emotions in RL Agents](https://arxiv.org/pdf/1705.05172.pdf)
    - [Social Influence in Multi-Agent RL](https://arxiv.org/pdf/1810.08647.pdf)
    - [Open AI Learning to Cooperate](https://openai.com/blog/learning-to-cooperate-compete-and-communicate/)
    - [Open AI Learning to Communicate](https://openai.com/blog/learning-to-cooperate-compete-and-communicate/)
    - [Joy, Destress, Hope, Fear in RL](https://dl.acm.org/doi/10.5555/2615731.2616089O)
    - [DDPG Paper](https://arxiv.org/abs/1509.02971)


    



