
python version 3.10

```bash
pip install scipy==1.11.1
pip install numpy==1.25.1
pip install matplotlib==3.7.2
pip install gym==0.26.2
pip install pyglet==2.0.8
pip install pyyaml
pip install wandb
```

Task : four in a row (9 x 4)

### why
9 x 4으로 한다면 어느 한쪽은 반드시 이긴다고 하심
비기는 경우가 절대 안나온다고 하셨음


### Todo
agent를 흑백으로 따로 둘 것
흑은 학습하고 백은 학습하지 않고
둘다 학습하고 
뭐



### 잘 모르겠는 것
fiar_env에서 225번째 줄 활성화하면 왜 invalid_channel에서 에러가 발생하는지 잘 모르겠음
코드 상 아군의 그룹과 적군의 그룹을 분석해서 적절한 움직임을 하겠다 라는거 같은데 
일단 보류
state[INVD_CHNL] = state_utils.compute_invalid_moves(state, player, ko_protect)

