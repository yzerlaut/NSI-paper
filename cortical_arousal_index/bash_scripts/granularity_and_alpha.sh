# the impact of alpha on the granularity of the classification
python exp_data.py -gpl -f data/final_depol_alpha1.npz --parallelize --alpha 1.
python exp_data.py -gpl -f data/final_depol_alpha2.npz --parallelize --alpha 2.
python exp_data.py -gpl -f data/final_depol_alpha3.npz --parallelize --alpha 3.
python exp_data.py -gpl -f data/final_depol_alpha4.npz --parallelize --alpha 4.
python exp_data.py -gpl -f data/final_depol_alpha5.npz --parallelize --alpha 5.
