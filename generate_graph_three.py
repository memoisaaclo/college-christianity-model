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

    # Create a figure with subplots: 2 rows (for elder efficacies) x 3 columns (for A matrices)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Store all results for consistent colorbar scale
    all_results = []

    # First pass: compute all results to find global min/max
    for idx_efficacy, B_step in enumerate(elder_efficacies):
        for idx_A, A in enumerate(As):
            results = np.zeros((steps, steps))
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
            all_results.append(results)

    # Find global min and max for consistent color scale
    vmin = min(np.min(r) for r in all_results)
    vmax = max(np.max(r) for r in all_results)

    # Second pass: plot all results with consistent colorbar
    result_idx = 0
    for idx_efficacy, B_step in enumerate(elder_efficacies):
        for idx_A, A in enumerate(As):
            # Get the appropriate subplot
            ax = axes[idx_efficacy, idx_A]

            results = all_results[result_idx]
            result_idx += 1

            results = np.flipud(results)

            sb.heatmap(results, cmap='viridis', cbar=False,
                       ax=ax, vmin=vmin, vmax=vmax)

            tick_positions = np.linspace(0, steps-1, 6)
            tick_labels = np.round(np.linspace(start, end, 6), 2)

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(np.flip(tick_labels))

            if idx_efficacy == 1:  # Only add x-labels to bottom row
                ax.set_xlabel('Beta Ret (p_CS, p_DS)')
            if idx_A == 0:  # Only add y-labels to leftmost column
                ax.set_ylabel('Beta (p_SC, p_SD)')

            ax.set_title(f'{A_labels[idx_A]}, Elder Efficacy {B_step}')

    # colorbar
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.15, 0.95, 0.7, 0.02])
    sm = plt.cm.ScalarMappable(
        cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Final C Percentage')

    plt.subplots_adjust(top=0.9)  # Adjusted for colorbar and suptitle
    plt.show()
