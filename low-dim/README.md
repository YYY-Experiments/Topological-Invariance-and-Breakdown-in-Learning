# Experiments with Low Dimensional Networks

This folder contains experiment code for the low-dimensional networks. For each experiment, run `main.py` to generate the experiment result and experiment id, and run `python -m analyze.draw_weights --exp_id=<exp_id>` or `python -m analyze.draw_weights_3d --exp_id=<exp_id>` to draw corresponding figures. 

Below are specific commands for each setting. 

### 2d networks
Run experiments.
```sh
# GD,  small lr
python main.py --num_epoch=100 --data=teacher --wd=0 --lr=0.0025 --hidden_size=1024

# GD, large lr 
python main.py --num_epoch=100 --data=teacher --wd=0 --lr=0.003 --hidden_size=1024

# GD, small lr, with altered initialization
python main.py --lr=0.0025 --data=teacher --hidden_size=1024 --alt --num_epoch=100

# GD, large lr, with altered initialization
python main.py --lr=0.003 --data=teacher --hidden_size=1024 --alt --num_epoch=100

# Adam, small lr
python main.py --data=teacher --hidden_size=1024 --optimizer=adam --lr=1e-4 --alt --num_epoch=40

# Adam large lr
python main.py --data=teacher --hidden_size=1024 --optimizer=adam --lr=1e-2 --alt --num_epoch=40

# Momentum, small lr
python main.py --num_epoch=50 --data=teacher --wd=0 --lr=0.0045 --hidden_size=1024 --optimizer=momentum

# Momentum, large lr
python main.py --num_epoch=50 --data=teacher --wd=0 --lr=0.0005 --hidden_size=1024 --optimizer=momentum
```

Generate figures. Replace `<exp_id>` with specific experiment ids generated in `python main.py` commands.
```sh
python -m analyze.draw_weights --exp_id=<exp_id> --frame_prop=0.
python -m analyze.draw_weights --exp_id=<exp_id> --frame_prop=1.
```

### 3d networks
```sh
# GD, small lr
python main.py --num_epoch=20 --data=teacher_3d --wd=0 --lr=0.0009 --hidden_size=4096

# GD, large lr
python main.py --num_epoch=20 --data=teacher_3d --wd=0 --lr=0.0008 --hidden_size=4096

# Adam, small lr
python main.py --lr=0.03 --data=teacher_3d --hidden_size=4096 --num_epoch=20 --optimizer=adam

# Adam, large lr
python main.py --lr=0.1 --data=teacher_3d --hidden_size=4096 --num_epoch=20 --optimizer=adam

# Momentum, small lr
python main.py --num_epoch=20 --data=teacher_3d --wd=0 --lr=0.001 --hidden_size=4096 --optimizer=momentum

# Momentum, large lr
python main.py --num_epoch=20 --data=teacher_3d --wd=0 --lr=0.00125 --hidden_size=4096 --optimizer=momentum
```
Generate figures. Replace `<exp_id>` with specific experiment ids generated in `python main.py` commands. Replace `<azim>`, `<elev>`, `<zoom>` with specific values to control the camera angle etc.
```sh
python -m analyze.draw_weights_3d --exp_id=<exp_id> --frame_prop=0. --azim=<azim> --elev=<elev> --zoom=<zoom>
python -m analyze.draw_weights_3d --exp_id=<exp_id> --frame_prop=1. --azim=<azim> --elev=<elev> --zoom=<zoom>
```

