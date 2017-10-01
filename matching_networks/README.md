# MatchingNetworks Tensorflow Implementation
This repo provides code that replicated the results of the Matching Networks for One Shot Learning paper on the Omniglot dataset.

To train a model
### Training a model
```bash
python train_one_shot_learning_matching_network.py --logs-path 20-shot/pool-1 --pool-range 1 # val_loss 0.169, val_accuracy 94.8
python train_one_shot_learning_matching_network.py --logs-path 20-shot/pool-2 --pool-range 2 # val_loss 0.147, val_accuracy 95.6
python train_one_shot_learning_matching_network.py --logs-path 20-shot/pool-3 --pool-range 3 # val_loss 0.151, val_accuracy 95.3
python train_one_shot_learning_matching_network.py --logs-path 20-shot/pool-4 --pool-range 4 # val_loss 0.144, val_accuracy 95.5
```

### To access all command line arguments
```bash
python train_one_shot_learning_matching_network.py -h
```
