# Compare speed of pytorch on m1 chips

## Hardware
device: M1(MacMini) \
device: M1 Ultra(MacStudio) \
device: Intel Xeon Gold 5218R(20 hyper-thread used) + RTX3090 


## Models

MLP: 1024 * 512 * 512 * 512 * 2, run for 10000 steps of batch size 128\
Resnet50: run for 50 steps of batch size 32

## Compare of inference time
| chips           | device | model     | time cost(seconds)   |   
| :-----------:   | :----- | :-----    | :---------:          |
| M1              |  GPU   |  MLP      |  4.8                 |
| M1 Ultra        |  GPU   |  MLP      |  5.1                 |
| RTX3090         |  GPU   |  MLP      |  3.3                 |
| M1              |  CPU   |  MLP      |  12.6                |
| M1 Ultra        |  CPU   |  MLP      |  14.8                |
| Xeon Gold       |  CPU   |  MLP      |  330                 |
| M1              |  GPU   |  Resnet50 |  16.9                | 
| M1 Ultra        |  GPU   |  Resnet50 |  2.7                 | 
| RTX3090         |  GPU   |  Resnet50 |  2.9                 |  
| M1              |  CPU   |  Resnet50 |  206                 | 
| M1 Ultra        |  CPU   |  Resnet50 |  210                 |  
| Xeon Gold       |  CPU   |  Resnet50 |  57                  | 

&nbsp;
## Compare of switch and no-switch
switch: data is copied from cpu to gpu every iteration before doing model forward(each batch only one switch happened)\
no-switch: all operations are done on gpu

| chips           | switch | model     | time cost(seconds)   |   
| :-----------:   | :----: | :-----    | :---------:          |
| M1              |  yes   |  MLP      |  9.1                 |
| M1              |  no    |  MLP      |  4.8                 |
| M1 Ultra        |  yes   |  MLP      |  13.0                |
| M1 Ultra        |  no    |  MLP      |  5.1                 |
| RTX3090         |  yes   |  MLP      |  8.4                 |
| RTX3090         |  no    |  MLP      |  3.3                 |
| M1              |  yes   |  Resnet50 |  17                  |
| M1              |  no    |  Resnet50 |  16.9                |
| M1 Ultra        |  yes   |  Resnet50 |  2.8                 |
| M1 Ultra        |  no    |  Resnet50 |  2.7                 |
| RTX3090         |  yes   |  Resnet50 |  4.5                 |
| RTX3090         |  no    |  Resnet50 |  2.9                 |

&nbsp;
## Compare of revive sdk
setting: venv algo: ppo, revive epoch: 50, ppo epoch: 20

| chips              | model        | time cost(seconds)   |   
| :-----------:      | :-----       | :---------:          |
| M1 Ultra(only cpu) |  revive      |  54                  |
| M1 Ultra(gpu)      |  revive      |  597                 |
| M1 (only cpu)      |  revive      |  65                  |
| RTX3090            |  revive      |  119                 |
| Xeon Gold          |  revive      |  200                 |

