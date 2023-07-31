# FQF, IQN, QR-DQN and DQN in PyTorch

This is a PyTorch implementation of Fully parameterized Quantile Function(FQF)[1], Implicit Quantile Networks(IQN)[2], Quantile Regression DQN(QR-DQN)[3] and Deep Q Network(DQN)[4].

python version 3.10

```bash
pip install scipy==1.11.1
pip install numpy==1.25.1
pip install matplotlib==3.7.2
pip install gym==0.26.2
pip install pyglet==2.0.8and
pip install pyyaml
pip install tensorboard
```

Task : four in a row (9 x 4)


### Todo

1. next_state에서 black이 둘때는 white는 -1로 만들고, white가 둘 때는 black을 -1로 만드는 식으로 state 들이 서로 겹치지 않도록
2. mat1의 1440 x 4를 수정할 방법을 알아보기
3. MCTS 알고리즘 수정
4. distribution RL 들끼리 비교도 좋지만 일반적인 DQN과 비교도 해보는 것과 planning을 했을 때 안했을 때 비교도 해보고 싶기 떄문에 
5. forward planning 방식 추가 (여기까지 하는걸 최우선으로)
6. hyperparameter 적용
7. 중복되는 action이 뽑히면 try except 문 사용해서 예외처리하기
