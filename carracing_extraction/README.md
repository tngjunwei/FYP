# CarRacing Experiments

Step by step instructions of reproducing CarRacing experiment.

# Notes

-TensorFlow 1.8.0

-NumPy 1.13.3 (1.14 has some annoying warning)

-OpenAI Gym 0.9.4 (have not tested 1.0+, big gym api changes)

-cma 2.2.0, basically 2+ should work

-Python 3, although 2 might work.

-mpi4py 2 (see https://github.com/hardmaru/estool)


# Instructions for training everything from scratch

Extract 10k random episodes by running the following on a 64-core CPU machine (note that since it uses OpenGL, it runs a headless X session for each worker job, which was needed in Ubuntu VMs):

`bash extract.bash`

After running this, 12.8k episodes should be saved as npz files in `record`. We will only use 10k episodes.

If it does not work, just do `python extract.py`