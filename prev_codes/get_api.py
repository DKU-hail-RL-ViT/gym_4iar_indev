import wandb
import pandas as pd

api = wandb.Api()

runs_info = [
    ("hails/gym_4iar_sh2/kcc396eh", "QRQAC_nmcts2_quantiles3.csv"),
    ("hails/gym_4iar_sh2/r16pg6cl", "QRQAC_nmcts10_quantiles27.csv"),
    ("hails/gym_4iar_sh2/tn49czn5", "QRQAC_nmcts50_quantiles27.csv"),
    ("hails/gym_4iar_sh2/36vpk2uc", "QRQAC_nmcts100_quantiles81.csv"),
    ("hails/gym_4iar_sh2/dp63mjgg", "QRQAC_nmcts400_quantiles81.csv"),
    ("hails/gym_4iar_sh2/kkbmgahl", "EQRQAC_nmcts2.csv"),
    ("hails/gym_4iar_sh2/znmzti9n", "EQRQAC_nmcts10.csv"),
    ("hails/gym_4iar_sh2/42y28jtw", "EQRQAC_nmcts50.csv"),
    ("hails/gym_4iar_sh2/m3cks2d9", "EQRQAC_nmcts100.csv"),
    ("hails/gym_4iar_sh2/e5xk5z8p", "EQRQAC_nmcts400.csv"),

]

for run_path, csv_name in runs_info:
    run = api.run(run_path)
    history = run.history()
    history.to_csv(csv_name, index=True)
