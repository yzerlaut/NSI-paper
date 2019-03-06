# the cross-validation script
python exp_data.py -tda -f data/final_alpha_first_third.npz --parallelize --discretization 40 --subsampling_period 1e-3 --alpha_min 1. --alpha_max 7. --data_fraction 0. 0.33
python exp_data.py -tda -f data/final_alpha_last_third.npz --parallelize --discretization 40 --subsampling_period 1e-3 --alpha_min 1. --alpha_max 7. --data_fraction 0.66 1.
