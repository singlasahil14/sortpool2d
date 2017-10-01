## loss-exps
Experiments with a different pooling layer for image classification

### Run cluttered mnist experiments
```bash
python train.py --result cluttered-mnist/pool_1 --pool-range 1 # val_cross_entropy 0.259
python train.py --result cluttered-mnist/pool_2 --pool-range 2 # val_cross_entropy 0.202
python train.py --result cluttered-mnist/pool_3 --pool-range 3 # val_cross_entropy 0.196
python train.py --result cluttered-mnist/pool_4 --pool-range 4 # val_cross_entropy 0.196
```

### Run fashion mnist experiments
```bash
python train.py --result fashion-mnist/pool_1 --pool-range 1 --dataset fashion-mnist # val_cross_entropy 0.291
python train.py --result fashion-mnist/pool_2 --pool-range 2 --dataset fashion-mnist # val_cross_entropy 0.282
python train.py --result fashion-mnist/pool_3 --pool-range 3 --dataset fashion-mnist # val_cross_entropy 0.268
python train.py --result fashion-mnist/pool_4 --pool-range 4 --dataset fashion-mnist # val_cross_entropy 0.276
```

### To access all command line arguments
```bash
python train.py -h
```
