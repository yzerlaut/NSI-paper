# first the wavelet scan
python exp_data.py -tda -f data/alpha40.npz --parallelize --discretization 40 --subsampling_period 1e-3 --args.alpha_min 1.5 --alpha_max 7.
