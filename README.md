# Explore This!

This is a fork of [Beat This!](https://github.com/CPJKU/beat_this). 

To run the code, follow the instructions for Beat This! about how to install all of the dependencies. 

Due to this project being a fork from Beat This from the start, many strange leftovers remain. Both in terms of names referencing Beat This and the pyproject.toml clearly including artifacts. In the interest of saving time while keeping reproducability, that has been left as is. Maybe in the future, I can get around to fixing it. 

Some things will be broken because I had to change the inference code. I made sure it worked for the final test, but other leftover references to inference stuff might be broken. 

## Training 
The cross validation models in the paper were trained using (i=0,1,...,7)
```
python launch_scripts/train.py --name=final_crossval --logger=wandb --transformer-dim=256 --n-layers=3 --subgrid-transformer-layers=3 --grid-window-size=50 --subgrid-regularization-scale=0.1 --grid-regularization-weight=0.1 --grid-reg-phase-factor=3 --grid-reg-freq-scale=0.004 --subgrid-max-downbeat-meter=7 --subgrid-loss-scale=30 --fold=i
```

The full version were trained with (i=0,1,2)
```
python launch_scripts/train.py --name=final_0 --logger=wandb --transformer-dim=256 --n-layers=3 --subgrid-transformer-layers=3 --grid-window-size=50 --subgrid-regularization-scale=0.1 --grid-regularization-weight=0.1 --grid-reg-phase-factor=3 --grid-reg-freq-scale=0.004 --subgrid-max-downbeat-meter=7 --subgrid-loss-scale=30 --no-val --seed=i
```

## Implementation of the grid and subgrid
This is the important part of this repository, and is what differs from Beat This. 
```beat_this/model/grid.py``` and ```beat_this/model/grid.py``` contain the majority of the logic for the modules, and the rest is either in files imported by those or in ```beat_this/model/beat_tracker.py``` (which chains together the modules) or ```beat_this/model/pl_module.py``` (where the losses are applied).

## Cite

```
@inproceedings{explorethis,
  author       = {Robert Kihlborg and
                  André Holzapfel and
                  Jan Schl{\"u}ter},
  title        = {Explore This! Beat and Downbeat Tracking From a Learned Tatum Grid},
  year         = {2026},
  booktitle    = {Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
}
```
