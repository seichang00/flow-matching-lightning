# Flow Matching for Conditional Image Generation
Implementation of flow matching-based conditional image generation on MNIST using Pytorch Lightning. 

## Install python packages

```
pip install -r requirements.txt
```

## Train flow matching model for MNIST

Data, training, and model parameters are set in the `setup.yaml` config file. To begin training on MNIST, run `trainer.py` within the source directory:

```
python trainer.py
```

## Running MNIST image generation

Model checkpoints are stored in the run directory, which is set as `{train.save_path}/{train.run_name}`. The path variables can be configured in `setup.yaml`.

In `demo.ipynb`, replace the `model_path` variable with the path to your saved checkpoint. After running all the cells, the last cell should provide grids of MNIST generated outputs.

## Additional Notes

The inference code in `demo.ipynb` uses `torchdyn` library for simulating the learned vector field. Feel free to experiment with different ODE solvers (ex. `rk4`, `dopri5`) and guidance scales. 

This repo was designed to be minimal for educational purposes as a follow up to the tutorial notebooks in [https://diffusion.csail.mit.edu/](https://diffusion.csail.mit.edu/). For a more extensive library on conditional flow matching, please refer to [
TorchCFM](https://github.com/atong01/conditional-flow-matching).



## Reference
Code has been adapted from the notebooks provided in [https://diffusion.csail.mit.edu/](https://diffusion.csail.mit.edu/). 

```
  @misc{flowsanddiffusions2025,
    author       = {Peter Holderrieth and Ezra Erives},
    title        = {Introduction to Flow Matching and Diffusion Models},
    year         = {2025},
    url          = {https://diffusion.csail.mit.edu/}
  }
```