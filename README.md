# DK

```bash
pip install sb3-contrib
pip install 'stable-baselines3[extra]'
pip install torch
pip install gymnasium
pip install scipy
pip install pyglet
```

# SH

python version 3.10

```bash
pip install sb3-contrib
pip install 'stable-baselines3[extra]'
pip install torch
pip install gymnasium
pip install pyglet
pip install scipy==1.11.1
pip install numpy==1.25.1
pip install matplotlib==3.7.2
pip install gym==0.26.2
pip install pyglet==2.0.8
pip install mcts
pip install wandb
```

Task : four in a row (9 x 4)


### Problem

교수님이 해결해주신 버전이라 문제 부분엔 적지 않았음

추가해야할건 루트 노드가 뭔지 나오게한다면 좋을거 같다고 하심

이건 고민해봐야 할 부분인데  winning에서 리턴 되는 값에 따라 누구를 학습시킬지가 결정이 되는데





### version  

교수님이 해결해주신 version
batch_size = 32   # previous 512 너무 오래걸려서 32로 줄여놓았음




black 학습, white 랜덤

black win -> reward : 1
white win -> reward : abs(-1)
draw -> reward : 0

player_myself = 0 , player_enemy = 1 일때 흑이 돌을 놓을 차례
player_myself = 1 , player_enemy = 0 일때 백이 돌을 놓을 차례

