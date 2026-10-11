# Explore This!

This is a fork of [Beat This!](https://github.com/CPJKU/beat_this). 

To run the code, follow the instructions for Beat This! about how to install all of the dependencies. 

This is a slightly cleaned up version of the code that was used for our paper, though it should hopefully not affect reproducability. 
If the cleaning accidentally broke something, the commit with hash 18906da44d2247b19829e0cf9a098bed6d42316f is the original. ¨


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
```explore_this/model/grid.py``` and ```explore_this/model/subgrid.py``` contain the majority of the logic for the modules, and the rest is either in files imported by those or in ```explore_this/model/beat_tracker.py``` (which chains together the modules) or ```explore_this/model/pl_module.py``` (where the losses are applied).

## Other differences
During developement, we played around with artificial data to test the isolated grid module, and also wrote some code to visualize and analyze certain parts of the model. This auxiliary code is kept in ```explore_this/architecture_testing``` for inspiration, but is not referenced by the actual model or the launch scripts. 

In ```notebooks/results.ipynb```, we include the script that calculates all paper-referenced metrics on the different datasets using the models trained with the above scripts. We used this instead of the "compute_paper_metrics.py" used by Beat This as it simplifies analysis. 

## Cite
```
@inproceedings{explorethis2026,
  author       = {Robert Kihlborg and
                  Andr{\´e} Holzapfel and
                  Jan Schl{\"u}ter},
  title        = {Explore This! Beat and Downbeat Tracking From a Learned Tatum Grid},
  year         = {2026},
  booktitle    = {Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
}
```