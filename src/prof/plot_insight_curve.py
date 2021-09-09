import matplotlib.pyplot as plt
import numpy as np 

def plot_curve():
    data = np.loadtxt('tmp/plot_insight/data.csv',delimiter=',')

    plt.rc('font', family='serif')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(6,2))

    x = np.array(list(range(1,13)))

    ax.plot(x, data[0], marker='o', label='all')
    ax.plot(x, data[1], marker='^', label='diagonal')

    plt.xticks(x)

    plt.xlabel('Layer')
    plt.ylabel('Accuracy')

    plt.legend(loc='lower right')

    plt.savefig('tmp/plot_insight/insight_accuracy.png', format='png',bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    plot_curve()