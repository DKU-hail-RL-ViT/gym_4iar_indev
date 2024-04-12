# SH
python version 3.10
Windows : torch version 2.2.2+cu118 
Mac OS M1: torch version 2.2.1


## References
### AlphaZero_Gomoku
https://github.com/junxiaosong/AlphaZero_Gomoku

### Elo rating system
https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details

### Requirements
https://escholarship.org/uc/item/8wm748d8


```bash
pip install torch
pip install gymnasium
pip install pyglet
pip install scipy==1.11.1
pip install numpy==1.25.1
pip install matplotlib==3.7.2
pip install gym==0.26.2
pip install pyglet==2.0.8
pip install wandb
```

Task : four in a row (9 x 4)

### Thinking
- How many quantiles should be used?
2, 4, 8, 16, 32, 64  ...?  (for c51)

- Distribution RL Network output (64, quantile) -> ex(64, 32)
so, loss and entropy are lager than AC model (complex calculation)

- Performance (AC model < QRAC model) 



### TODO
- train_fiar.py 233 line  (leaf node MCTS)
- train_fiar.py 234 line  (random actions Agent)



### version
- MCTS + RL
1) MCTS + AC model (default alphazero)
2) MCTS + QRAC model (Quantile Regression Actor Critic)
3) MCTS + EQRAC model (Efficient Quantile Regression Actor Critic)


Black: MCTS policy trained through the policy value network
White: MCTS policy based solely on pure MCTS

Black win -> Reward: 1
White win -> Reward: -5e-1
Draw -> Reward: -1


# adjust hyperparameter
### tuning parameter 
n_playout = 20  # = MCTS simulations(n_mcts) & training 2, 20, 50, 100, 400


### MCTS parameter
buffer_size = queue(36 * 20) # board size * last 20 games
c_puct = 5
epochs = 10  # During each training iteration, the DNN is trained for 10 epochs.
self_play_sizes = 100
temperature = 0.1


### Policy update parameter 
batch_size = 64 
learn_rate = 5e-4  # previous 2e-3
lr_mul = 1.0
lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
kl_targ = 0.02  


### Policy evaluate parameter 
win_ratio = 0.0
init_model = None



## Summary

the overall process can be broadly divided into the self-play and start-play phases. 

During self-play, a single Monte Carlo Tree Search (MCTS) alternately plays as both black and white, 
learning through a policy-value network composed of a Multi-Layer Perceptron (MLP), 
akin to an actor-critic model.

The MCTS trained during self-play becomes MCTS 1, and in subsequent games, MCTS 2 is set to choose actions 
purely through MCTS without passing through the policy-value network. 

In the start-play phase, MCTS 1 and MCTS 2, now representing black and white policies, engage in games. 
By default, MCTS 1 is configured to play as black.

The win rate of MCTS 1 during these games is used as a benchmark, and the corresponding policy version 
is saved if the win rate is higher. 
This saved policy version is then compared with the ongoing policies, allowing for the continuous refinement 
of policies based on their performance. 
