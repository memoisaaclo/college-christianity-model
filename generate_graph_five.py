import final_project_model

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("info: graphing (5)")

    beta = .40
    beta_ret = .20
    B_step = .1
    B = [[round(beta + B_step*(j - i), 3)
          for j in range(4)] for i in range(4)]
    A = [
        [.3, .3, .25, .15],
        [.3, .3, .25, .15],
        [.3, .3, .25, .15],
        [.3, .3, .25, .15]
    ]
    parameters = {
        'p_SC': beta,
        'p_CS': beta_ret,
        'p_SD': beta,
        'p_DS': beta_ret,
        'A': A,
        'B': B,
    }

    results = np.zeros((100, 100))
    restocks = np.linspace(0, .5, 100)
    for i, C_restock in enumerate(restocks):
        for j, D_restock in enumerate(restocks):
            S_restock = 1 - C_restock - D_restock
            initial_conditions = {
                'C_1': C_restock,
                'S_1': S_restock,
                'D_1': D_restock,

                'C_2': C_restock,
                'S_2': S_restock,
                'D_2': D_restock,

                'C_3': C_restock,
                'S_3': S_restock,
                'D_3': D_restock,

                'C_4': C_restock,
                'S_4': S_restock,
                'D_4': D_restock,

                'C_incoming': C_restock,
                'S_incoming': S_restock,
                'D_incoming': D_restock,
            }

            model = final_project_model.DiscreteReligiousBeliefModel(
                parameters, initial_conditions, simulation_years=10)
            model.run_simulation()

            C_percent = np.sum(
                model.results[-1][:, final_project_model.C_IDX])/4
            results[i][j] = C_percent

    # put 0, 0 in the bottom left
    results_flipped = np.flipud(results)

    plt.figure(figsize=(10, 8))
    ax = sb.heatmap(results_flipped, cmap='viridis', cbar_kws={'label': 'Final C Percentage'})

    # fix tick positions
    tick_positions = np.linspace(0, 99, 6)
    tick_labels = np.round(np.linspace(0, 0.5, 6), 2)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(np.flip(tick_labels))

    ax.set_xlabel('D Restock Rate')
    ax.set_ylabel('C Restock Rate')
    plt.title('Final C Percentage by Restock Rates')

    plt.tight_layout()
    plt.show()
