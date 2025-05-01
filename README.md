# Flow Matching for Conditional Image Generation
Implementation of flow matching on MNIST using Pytorch Lightning. 

## Install python packages

```
pip install -r requirements.txt
```

## Train flow matching model for MNIST

Data, training, and model parameters are set in the `setup.yaml` config file. To begin training on MNIST, run `trainer.py` within the source directory.


## Reference
Code has been adapted from the notebooks provided in [https://diffusion.csail.mit.edu/](https://diffusion.csail.mit.edu/).