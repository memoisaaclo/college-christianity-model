import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


if __name__ == "__main__":
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
    beta = 0.4

    A_titles = ['homophily', 'heterophily', 'STA data']
    B_titles = ['homogenus transmission', 'heterogenus transmission']
    tick_labels = np.linspace(3, 0, 4, dtype=int)
    for i, A in enumerate(As):
        ax = sb.heatmap(np.flipud(A), cmap='viridis', cbar_kws={'label': 'contact density'})
        plt.title('A contact matrix - ' + A_titles[i])
        ax.set_yticklabels(tick_labels)
        plt.show()
    for i, B_step in enumerate(elder_efficacies):
        B = [[round(beta + B_step*(j - i), 3) for j in range(4)] for i in range(4)]
        ax = sb.heatmap(np.flipud(B), cmap='viridis', cbar_kws={'label': 'transmission rate'})
        plt.title('B transmission matrix - ' + B_titles[i])
        ax.set_yticklabels(tick_labels)
        plt.show()
