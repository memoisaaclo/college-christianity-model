import final_project_model
import pprint as pp

if __name__ == "__main__":
    print("info: graphing (1)")
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

    # fix betas
    beta = 0.4
    beta_ret = 1.0
    parameters = {
        'p_SC': beta,
        'p_CS': beta_ret,
        'p_SD': beta,
        'p_DS': beta_ret,
    }

    for A in As:
        for B_step in elder_efficacies:
            B = [[round(beta + B_step*(j - i), 3) for j in range(4)] for i in range(4)]
            print("B:")
            pp.pprint(B)

            print("A:")
            pp.pprint(A)

            parameters['A'] = A
            parameters['B'] = B

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

        model = final_project_model.DiscreteReligiousBeliefModel(parameters, initial_conditions, simulation_years=10)
        model.run_simulation()
        model.plot_results()
