import pprint as pp
import numpy as np
import matplotlib.pyplot as plt

C_IDX = 0
S_IDX = 1
D_IDX = 2


class DiscreteReligiousBeliefModel:
    def __init__(self, params, initial_conds, simulation_years=4):
        self.params = params
        self.simulation_years = simulation_years
        self.num_age_groups = 4

        # create initial state matrix [age_group, compartment]
        # compartments in order C, S, D
        state = np.zeros((self.num_age_groups, 3))

        # initialize first generation
        for i in range(self.num_age_groups):
            age_idx = i + 1

            c_val = initial_conds.get(f'C_{age_idx}', 1 / 3)
            s_val = initial_conds.get(f'S_{age_idx}', 1 / 3)
            d_val = initial_conds.get(f'D_{age_idx}', 1 / 3)

            state[i, C_IDX] = c_val
            state[i, S_IDX] = s_val
            state[i, D_IDX] = d_val

        # normalize the state matrix once
        row_sums = np.sum(state, axis=1, keepdims=True)
        state = state / row_sums

        self.state = state
        self.initial_conditions = initial_conds

        self.results = np.zeros((simulation_years + 1, self.num_age_groups, 3))
        self.results[0] = self.state.copy()

    def run_simulation(self):
        p_CS = self.params.get('p_CS', 0.05)
        p_SC = self.params.get('p_SC', 0.05)
        p_DS = self.params.get('p_DS', 0.05)
        p_SD = self.params.get('p_SD', 0.05)
        A = np.array(self.params.get('A', [[0.25, 0.25, 0.25, 0.25] for _ in range(self.num_age_groups)]))
        B = np.array(self.params.get('B', [[p_SC for _ in range(self.num_age_groups)] for _ in range(self.num_age_groups)]))

        # incoming distribution
        fresh_C = self.initial_conditions.get('C_incoming', 1 / 3)
        fresh_S = self.initial_conditions.get('S_incoming', 1 / 3)
        fresh_D = self.initial_conditions.get('D_incoming', 1 / 3)
        total = fresh_C + fresh_S + fresh_D
        incoming = np.array([fresh_C/total, fresh_S/total, fresh_D/total])

        for year in range(1, self.simulation_years + 1):
            new_state = np.zeros_like(self.state)

            # move students up a year
            new_state[1:] = self.state[:-1]
            new_state[0] = incoming

            # belief transitions
            C = new_state[:, C_IDX]
            S = new_state[:, S_IDX]
            D = new_state[:, D_IDX]

            combined_influence = A * B
            S_to_C = S * np.dot(combined_influence, C)
            C_to_S = C * D * S * p_CS
            S_to_D = D * S * p_SD
            D_to_S = C * D * S * p_DS

            # update
            new_state[:, C_IDX] = C - C_to_S + S_to_C
            new_state[:, S_IDX] = S - S_to_C - S_to_D + C_to_S + D_to_S
            new_state[:, D_IDX] = D - D_to_S + S_to_D

            # no negatives from floating point
            new_state = np.maximum(new_state, 0)

            # normalize in a single operation
            row_sums = np.sum(new_state, axis=1, keepdims=True)
            new_state = new_state / row_sums

            # update and store the state
            self.state = new_state
            self.results[year] = self.state.copy()

        return self.results

    def plot_results(self):
        states = self.results
        years = np.arange(len(states))

        # mean across age groups
        # assume equal cohort sizes
        total_C = np.mean(states[:, :, 0], axis=1)
        total_S = np.mean(states[:, :, 1], axis=1)
        total_D = np.mean(states[:, :, 2], axis=1)

        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        plt.plot(years, total_C, 'b-', label='Christian (C)')
        plt.plot(years, total_S, 'g-', label='Susceptible (S)')
        plt.plot(years, total_D, 'r-', label='Denying (D)')
        plt.xlabel('Years')
        plt.ylabel('Population Proportion')
        plt.title('Total Population Proportions by Compartment')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.0)

        compartments = ['Christian (C)', 'Susceptible (S)', 'Denying (D)']
        colors = ['b', 'g', 'r', 'y']

        for c_idx, compartment in enumerate(compartments):
            plt.subplot(2, 2, c_idx + 2)
            for age in range(self.num_age_groups):
                plt.plot(years, states[:, age, c_idx], f'{colors[age]}-',
                         label=f'{compartment} - Year {age + 1}')
            plt.xlabel('Years')
            plt.ylabel('Population Proportion')
            plt.title(f'{compartment} by Academic Year')
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 1.0)

        plt.tight_layout()
        plt.show()

        # stacked area chart
        plt.figure(figsize=(12, 6))
        plt.stackplot(years, total_C, total_S, total_D,
                      labels=['Christian (C)', 'Susceptible (S)', 'Denying (D)'],
                      colors=['b', 'g', 'r'], alpha=0.7)
        plt.xlabel('Years')
        plt.ylabel('Population Proportion')
        plt.title('Evolution of Religious Belief Proportions')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    B_step = 0.1
    beta = 0.4
    beta_ret = beta * 2
    # for beta 0.20 and B_step 0.025 B looks like:
    # [[0.2, 0.225, 0.25, 0.275],
    #  [0.175, 0.2, 0.225, 0.25],
    #  [0.15, 0.175, 0.2, 0.225],
    #  [0.125, 0.15, 0.175, 0.2]]
    # Influence rate on i from j, B[i][j]
    B = [[round(beta + B_step*(j - i), 3) for j in range(4)] for i in range(4)]
    print("B:")
    pp.pprint(B)

    homophily = .4
    heterophily = round((1 - homophily) / 3, 3)
    # for homophily .1 A looks like:
    # [[0.1, 0.3, 0.3, 0.3],
    #  [0.3, 0.1, 0.3, 0.3],
    #  [0.3, 0.3, 0.1, 0.3],
    #  [0.3, 0.3, 0.3, 0.1]]
    A = [[homophily if i == j else heterophily for j in range(4)] for i in range(4)]
    print("A:")
    pp.pprint(A)

    parameters = {
        'p_SC': beta,
        'p_CS': beta_ret,
        'p_SD': beta,
        'p_DS': beta_ret,
        'A': A,
        'B': B
    }

    initial_conditions = {
        'C_1': 0.2,
        'S_1': 0.6,
        'D_1': 0.2,

        'C_2': 0.25,
        'S_2': 0.5,
        'D_2': 0.25,

        'C_3': 0.3,
        'S_3': 0.4,
        'D_3': 0.3,

        'C_4': 0.35,
        'S_4': 0.3,
        'D_4': 0.35,

        'C_incoming': 0.3,
        'S_incoming': 0.4,
        'D_incoming': 0.3,
    }

    model = DiscreteReligiousBeliefModel(parameters, initial_conditions, simulation_years=10)
    model.run_simulation()
    model.plot_results()
