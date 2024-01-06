import subprocess
import pandas as pd
import numpy as np
import os
import time
import csv



headers = ['H1_freq', 'spinning_freq' ,'power_1_1_1' ,'cs_beta_2', 'ss_iso_1_2', 'ss_ani_1_2', 'ss_beta_1_2', 'ss_ani_1_3', 'ss_beta_1_3', 'mw', 'T1e1']
ranges = [(50,1300),(0.05,80),(0,1500),(0,180),(-600e3,600e3),(-1200e3,1200e3),(0,180),(0,50e3),(0,180),(-1200e3,1200e3),(0.01,50)]

# Run the spinnev command
os.system(f"spinev Tarek")

with open('input.txt', 'r') as f1:
    next(f1)  # Skip the header
    data1 = [line.strip().split() for line in f1]

# Open the second file and read in the data
with open('Tarek_re.dat', 'r') as f2:
    data2 = [line.strip().split()[1:] for line in f2]

combined_data = [d1 + d2 for d1, d2 in zip(data1, data2)]

with open('combined.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerows(combined_data)









