# FQF, IQN, QR-DQN and DQN in PyTorch

This is a PyTorch implementation of Fully parameterized Quantile Function(FQF)[1], Implicit Quantile Networks(IQN)[2], Quantile Regression DQN(QR-DQN)[3] and Deep Q Network(DQN)[4].

python version 3.10

```bash
pip install scipy==1.11.1
pip install numpy==1.25.1
pip install matplotlib==3.7.2
pip install gym==0.26.2
pip install pyglet==2.0.8
pip install mcts
pip install wandb
```

Task : four in a row (9 x 4)


### Todo

model.qrdqn 3번 kill함 
지금 코드가 mcts만 돌아가도록 만들었음
qrdqn 다 빼놓음
1. 이제 해야할게 mcts에서 받아온 승률 가지고 qrdqn를 해야하지 않나
2. 아니면 CNN 끝나자마자 코드를 QRDQN에게 줘서 강화학습을 하거나 




