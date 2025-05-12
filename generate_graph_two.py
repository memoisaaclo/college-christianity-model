import final_project_model

import pprint as pp
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    restock = 1/3
    initial_conditions = {
        'C_1': restock,
        'S_1': restock,
        'D_1': restock,

        'C_2': restock,
        'S_2': restock,
        'D_2': restock,

        'C_3': restock,
        'S_3': restock,
        'D_3': restock,

        'C_4': restock,
        'S_4': restock,
        'D_4': restock,

        'C_incoming': restock,
        'S_incoming': restock,
        'D_incoming': restock,
    }

    print("info: graphing (2)")
    B_step = .1
    # STA data
    A = [
            [.3, .3, .25, .15],
            [.3, .3, .25, .15],
            [.3, .3, .25, .15],
            [.3, .3, .25, .15]
        ]

    betas = np.linspace(.01, 4, 100)
    beta_rets = np.linspace(.01, 4, 100)

    results = np.zeros((100, 100))
    for i, beta in enumerate(betas):
        for j, beta_ret in enumerate(beta_rets):
            B = [[round(beta + B_step*(j - i), 3)
                  for j in range(4)] for i in range(4)]
            parameters = {
                'p_SC': beta,
                'p_CS': beta_ret,
                'p_SD': beta,
                'p_DS': beta_ret,
                'A': A,
                'B': B,
            }

            model = final_project_model.DiscreteReligiousBeliefModel(
                parameters, initial_conditions, simulation_years=10)
            model.run_simulation()

            C_percent = np.sum(model.results[-1][:, final_project_model.C_IDX])/4
            results[i][j] = C_percent

    sb.heatmap(results, cmap='viridis')
    plt.show()
