import numpy as np
import pickle
from utils import *


with open('resources/results', 'rb') as f:
    functions_final_fitness = pickle.load(f)

print(functions_final_fitness.shape)
function_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12']
algo_names = ['PSO', 'CMA-ES', 'DE']
seed_names = [f'Seed {i + 1}' for i in range(30)]

cec_2022 = [pd.DataFrame(functions_final_fitness[i, :, :]).transpose()
                for i in range(functions_final_fitness.shape[0])]
for df in cec_2022:
    df.columns = algo_names
    df.index = seed_names

print(cec_2022[0])

compile_and_boxplot(functions_final_fitness, True)
