# Formulating Emotions in Reinforcement Learning Agents

## [Ideas File](https://docs.google.com/document/d/1c3yM_woKnLbukBI9sGDSJJ4zpply-S0q2qNzZN6L4VE/edit)
## [Markdown Cheat-Sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
## TODO 
- Add Emotion Functions to Model 
    - Formulate emotions based off core RL features in Algo files
        - reward function (incorporate emotions?)
        - emotion functions (from types above)
- Content Theory / Main webpage README
    - Write Wiki on Reinforcement Learning / DDPG Algo 
    - Connect concepts to content theory 
    - Explain SIM w/ Video Examples
- Establish process to run on GPU clusters
- Setup instructions to run in Colab
- README for running sims


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


    



