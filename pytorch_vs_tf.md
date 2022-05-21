# Compare speed of pytorch and tensorflow on m1 chips

## Hardware
machine: Macmini9, chip: M1

## Compare of model MLP (1024 * 512 * 512 * 2)

#### measurement: running for 10000 steps of batch size 32
| backend         | time cost   |  gpu usage 
| :-----------:   | :---------: |:---------:
| pytorch gpu     | 9.4 seconds | 100% 720MHZ
| pytorch vulkan  | 28 seconds  | 100% 720MHZ 
| pytorch cpu     | 3.6 seconds | 0% 
| tensorflow gpu  | 16 seconds  | 100% 720MHZ 
| tensorflow cpu  | 7.8 seconds | 0% 

&nbsp;
## Compare of model Resnet50

#### measurement: running for 50 steps of batch size 32

| backend         | time cost   |  gpu usage |
| :---------:     | :---------: |:---------:|
| pytorch gpu     | 17 seconds  | 100% 1200MHZ |
| pytorch cpu     | 44 seconds  | 0% |
| tensorflow gpu  | 17.3 seconds| 100% 1200MHZ |
| tensorflow cpu  | 54 seconds  | 0% |
