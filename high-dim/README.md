# Experiments with Practical Networks

This folder contains experiment code for the networks on real datasets. This codes depends on python packages `hessian_eigenthings` and `gudhi`.

For each experiment, run `python main.py` to generate the experiment result and corresponding experiment id, and run `python -m analyze.draw` to draw Betti number curves or `python -m analyze.draw_kinv` to draw sharpness inversion curves. See below for detailed instructions.

Below are specific commands for each setting. 

### Specific Commands

Generate experiment results. Replace `<seed>` with specific random seed.
```sh

# GD small lr
python main.py --lr=0.01 --bs=1024 --hidden_size=1024 --num_epoch=20 --seed=<seed>

# GD large lr
python main.py --lr=0.5 --bs=1024 --hidden_size=1024 --num_epoch=20 --seed=<seed>

# Adam small lr
python main.py --lr=1e-5 --bs=1024 --hidden_size=1024 --num_epoch=40 --optimizer=adam --seed=<seed>

# Adam large lr
python main.py --lr=0.001 --bs=1024 --hidden_size=1024 --num_epoch=20 --optimizer=adam --seed=<seed>
```

Draw figures. Replace `<exp_id_k>` and `<inset_exp_id_k>` with specific experiment ids generated when running `python main.py`. The experiment ids should be separated with a comma `,`. Make sure the experiments are ran in the same setting only with different random seeds. `<inset_exp_id_k>` is used as the experiment id for the inset figures.
```sh
python -m analyze.draw --exp_ids=<exp_id_1>,<exp_id_2>,<...>,<exp_id_n> --inset_exp_ids=<inset_exp_id_1>,<inset_exp_id_2>,<...>,<inset_exp_id_n>
python -m analyze.draw_kinv --exp_ids=<exp_id_1>,<exp_id_2>,<...>,<exp_id_n>
```