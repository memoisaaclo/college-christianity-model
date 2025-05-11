import numpy as np
import matplotlib.pyplot as plt


class DiscreteReligiousBeliefModel:
    def __init__(self, params, initial_conds, simulation_years=4):
        # validate params
        for param, value in params.items():
            if param.startswith('p_') and (value < 0 or value > 1):
                raise ValueError(f"Probability {param} must be between 0 and 1")

        self.params = params
        self.simulation_years = simulation_years
        self.num_age_groups = 4

        # create initial state matrix [age_group, compartment]
        state = np.zeros((self.num_age_groups, 3))

        # initialize state
        for i in range(self.num_age_groups):
            age_idx = i + 1

            c_val = initial_conds.get(f'C_{age_idx}', 1 / 3)
            s_val = initial_conds.get(f'S_{age_idx}', 1 / 3)
            d_val = initial_conds.get(f'D_{age_idx}', 1 / 3)

            state[i, 0] = c_val
            state[i, 1] = s_val
            state[i, 2] = d_val

            row_sum = np.sum(state[i])
            if row_sum > 0:
                state[i] = state[i] / row_sum

        self.state = state
        self.initial_conditions = initial_conds

        self.results = {
            'years': np.arange(simulation_years + 1),
            'states': np.zeros((simulation_years + 1, self.num_age_groups, 3))
        }
        self.results['states'][0] = self.state.copy()

    def run_simulation(self):
        p_CS = self.params.get('p_CS', 0.05)
        p_SC = self.params.get('p_SC', 0.05)
        p_DS = self.params.get('p_DS', 0.05)
        p_SD = self.params.get('p_SD', 0.05)
        alpha = self.params.get('alpha', [[0.25, 0.25, 0.25, 0.25] for _ in range(self.num_age_groups)])

        for year in range(1, self.simulation_years + 1):
            new_state = np.zeros_like(self.state)

            # move students up a year
            for i in range(1, self.num_age_groups):
                new_state[i] = self.state[i - 1]

            # incoming distribution
            fresh_C = self.initial_conditions.get('C_incoming',
                                                  self.initial_conditions.get('C_1', 1 / 3))
            fresh_S = self.initial_conditions.get('S_incoming',
                                                  self.initial_conditions.get('S_1', 1 / 3))
            fresh_D = self.initial_conditions.get('D_incoming',
                                                  self.initial_conditions.get('D_1', 1 / 3))

            # normalize
            total = fresh_C + fresh_S + fresh_D
            new_state[0, 0] = fresh_C / total
            new_state[0, 1] = fresh_S / total
            new_state[0, 2] = fresh_D / total

            # belief transitions
            for i in range(self.num_age_groups):
                # current
                C = new_state[i, 0]
                S = new_state[i, 1]
                D = new_state[i, 2]

                # stubbornness factor
                stub = 1.00**i

                C_to_S = C * p_CS * stub
                S_to_C = 0
                for j in range(self.num_age_groups):
                    S_to_C += S * new_state[j, 0] * alpha[i][j] * p_SC

                S_to_D = S * p_SD
                D_to_S = D * p_DS * stub

                # update
                new_state[i, 0] = C - C_to_S + S_to_C  # Christian
                new_state[i, 1] = S - S_to_C - S_to_D + C_to_S + D_to_S  # Susceptible
                new_state[i, 2] = D - D_to_S + S_to_D  # Denying

                # no negatives due from floating point
                new_state[i] = np.maximum(new_state[i], 0)

                new_state[i] = new_state[i] / np.sum(new_state[i])

            # Update the state
            self.state = new_state

            # Store the state for this year
            self.results['states'][year] = self.state.copy()

        return self.results

    def plot_results(self):
        years = self.results['years']
        states = self.results['states']

        # mean across age groups
        # assume equal cohort sizes
        total_C = np.mean(states[:, :, 0], axis=1)
        total_S = np.mean(states[:, :, 1], axis=1)
        total_D = np.mean(states[:, :, 2], axis=1)

        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        plt.plot(years, total_C, 'b-o', label='Christian (C)')
        plt.plot(years, total_S, 'g-o', label='Susceptible (S)')
        plt.plot(years, total_D, 'r-o', label='Denying (D)')
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
                plt.plot(years, states[:, age, c_idx], f'{colors[age]}-o',
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
    parameters = {
        'p_CS': 0.05,
        'p_SC': 0.20,
        'p_DS': 0.05,
        'p_SD': 0.20,
        'alpha': [
            [0.5, 0.5/3, 0.5/3, 0.5/3],
            [0.5/3, 0.5, 0.5/3, 0.5/3],
            [0.5/3, 0.5/3, 0.5, 0.5/3],
            [0.5/3, 0.5/3, 0.5/3, 0.5]
        ],
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
    # look for single