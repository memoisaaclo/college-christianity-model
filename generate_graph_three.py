import final_project_model

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
    print("info: graphing (3)")
    elder_efficacies = [0, .1]
    As = [
        # case (1), homophily, looks like
        # .7 .1 .1 .1
        # .1 .7 .1 .1
        # .1 .1 .7 .1
        # .1 .1 .1 .7
        [[.7 if i == j else .1 for j in range(4)] for i in range(4)],
        # case (2), heterophily, looks like
        # .1 .3 .3 .3
        # .3 .1 .3 .3
        # .3 .3 .1 .3
        # .3 .3 .3 .1
        [[.1 if i == j else .3 for j in range(4)] for i in range(4)],
        # case (3), STA data, looks like
        [
            [.3, .3, .25, .15],
            [.3, .3, .25, .15],
            [.3, .3, .25, .15],
            [.3, .3, .25, .15]
        ]
    ]
    A_labels = ["Homophily", "Heterophily", "STA Data"]

    start, end, steps = 0, 1, 50
    betas = np.linspace(start, end, steps)
    beta_rets = np.linspace(start, end, steps)

    plt.figure(figsize=(10, 8))
    for idx_efficacy, B_step in enumerate(elder_efficacies):
        for idx_A, A in enumerate(As):
            results = np.zeros((steps, steps))

            # Run simulations for each beta and beta_ret combination
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
                    C_percent = np.sum(
                        model.results[-1][:, final_project_model.C_IDX])/4
                    results[i][j] = C_percent

            # put 0, 0 in the bottom left
            results = np.flipud(results)
            ax = sb.heatmap(results, cmap='viridis', cbar_kws={'label': 'Final C Percentage'})

            # Set the correct tick positions
            tick_positions = np.linspace(start, steps, 9)
            tick_labels = np.round(np.linspace(start, end, 9), 2)

            # Set and format the axis ticks and labels
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(np.flip(tick_labels))

            # Add axis labels
            ax.set_xlabel('Beta Ret (p_CS, p_DS)')
            ax.set_ylabel('Beta (p_SC, p_SD)')

            # Add a descriptive title
            plt.title(f'Final C Percentage: {A_labels[idx_A]} with Elder Efficacy {B_step}')

            plt.tight_layout()
            plt.show()
