import numpy as np
import matplotlib.pyplot as plt
import os

def plot_dual_subplots():
    junk = np.load('tmp/prof_attn/baseline_junk.npy')
    ans = np.load('tmp/prof_attn/baseline_ans.npy')

    plt.rc('font', family='serif')
    plt.style.use('ggplot')

    # layer_idx = 10
    for layer_idx in range(12):

        layer_junk = junk[layer_idx]
        layer_ans = ans[layer_idx]

        layer_ans = layer_ans[~np.isnan(layer_ans).any(axis=1)]

        layer_junk = layer_junk.transpose()
        layer_ans = layer_ans.transpose()


        print(layer_junk.shape, layer_ans.shape)

        layer_junk = [layer_junk[i] for i in range(layer_junk.shape[0])]
        layer_ans = [layer_ans[i] for i in range(layer_ans.shape[0])]


        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122, sharey=ax1)

        # rectangular box plot
        bplot1 = ax1.boxplot(layer_junk,
                            vert=True,  # vertical box alignment
                            patch_artist=True,
                            showfliers=False, whis=0)  # will be used to label x-ticks
        # ax1.set_title('irrelevant tokens')

        # notch shape box plot
        bplot2 = ax2.boxplot(layer_ans,
                            vert=True,  # vertical box alignment
                            patch_artist=True,
                            showfliers=False, whis=0)  # will be used to label x-ticks
        # ax2.set_title('ans tokens')

        # fill with colors
        colors = ['pink', 'lightblue', 'lightgreen', 'lightsalmon', 'lightyellow', 'lightcyan','pink', 'lightblue', 'lightgreen', 'lightsalmon', 'lightyellow', 'lightcyan']
        for bplot in (bplot1, bplot2):
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        # adding horizontal grid lines
        for ax in [ax1, ax2]:
            ax.yaxis.grid(True)
            ax.set_xlabel('head')
            ax.set_ylabel('average attention')

        # plt.show()
        fig_name = f'tmp/plot/prof_attn_distrib_{layer_idx}.png'

        if not os.path.exists(os.path.dirname(fig_name)):
            os.makedirs(os.path.dirname(fig_name))

        plt.savefig(fig_name, format='png')
        plt.close()

def plot_grouped():
    junk = np.load('tmp/prof_attn/baseline_junk.npy')
    ans = np.load('tmp/prof_attn/baseline_ans.npy')

    plt.rc('font', family='serif')
    plt.style.use('ggplot')

    for layer_idx in range(12):

        layer_junk = junk[layer_idx]
        layer_ans = ans[layer_idx]

        layer_ans = layer_ans[~np.isnan(layer_ans).any(axis=1)]

        layer_junk = layer_junk.transpose()
        layer_ans = layer_ans.transpose()


        print(layer_junk.shape, layer_ans.shape)

        layer_junk = [layer_junk[i] for i in range(layer_junk.shape[0])]
        layer_ans = [layer_ans[i] for i in range(layer_ans.shape[0])]


        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 2))
        # ax1 = plt.subplot(121)
        # ax2 = plt.subplot(122, sharey=ax1)

        # rectangular box plot

        x_position = np.array(range(len(layer_junk)))*2.0

        bplot1 = ax.boxplot(layer_junk,
                            positions=x_position-0.4,
                            vert=True,  # vertical box alignment
                            patch_artist=True,
                            showfliers=False, whis=0)  # will be used to label x-ticks
        # ax1.set_title('irrelevant tokens')

        # notch shape box plot
        bplot2 = ax.boxplot(layer_ans,
                            positions=x_position+0.4,
                            vert=True,  # vertical box alignment
                            patch_artist=True,
                            showfliers=False, whis=0)  # will be used to label x-ticks
        # ax2.set_title('ans tokens')


        # fill with colors
        # colors = ['pink', 'lightblue', 'lightgreen', 'lightsalmon', 'lightyellow', 'lightcyan','pink', 'lightblue', 'lightgreen', 'lightsalmon', 'lightyellow', 'lightcyan']
        for bplot, color in zip((bplot1, bplot2), ['lightsalmon', 'lightblue']):
            for patch in bplot['boxes']:
                patch.set_facecolor(color)

        plt.xticks(x_position, [f"{i}" for i in range(12)])


        ax.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ['Context', 'Answer'], loc='best')

        # # adding horizontal grid lines
        # for ax in [ax1, ax2]:
        #     ax.yaxis.grid(True)
        #     ax.set_xlabel('head')
        #     ax.set_ylabel('average attention')

        plt.xlabel('Head')
        plt.ylabel('Avg. attention value')

        plt.title(f"Layer {layer_idx}")

        # plt.show()
        fig_name = f'tmp/plot/attn/prof_attn_distrib_{layer_idx}.png'

        if not os.path.exists(os.path.dirname(fig_name)):
            os.makedirs(os.path.dirname(fig_name))

        plt.savefig(fig_name, format='png',bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    plot_grouped()