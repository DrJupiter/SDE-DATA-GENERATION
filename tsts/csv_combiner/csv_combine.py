import numpy as np
import pandas as pd
def combine_csv(path):
    csv = pd.read_csv(path, sep=',', names=["Step", "Loss", "Min Loss", "Max Loss"], header=0)

    return csv

print(combine_csv("./tmp/csv_combiner/combined_1.csv"))


