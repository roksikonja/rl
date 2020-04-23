import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


class Visualizer(object):
    @staticmethod
    def plot_signals(
        signal_triplets,
        results_dir=None,
        title=None,
        legend=None,
        y_lim=None,
        x_label=None,
        y_label=None,
    ):

        fig = plt.figure()
        for signal_triplet in signal_triplets:
            if not signal_triplet:
                continue

            t = signal_triplet[0]
            x = signal_triplet[1]
            label = signal_triplet[2]

            plt.plot(t, x, label=label)

        if title:
            plt.title(title)

        if legend:
            plt.legend(loc=legend)
        else:
            plt.legend()

        if y_lim:
            plt.gca().set_ylim(bottom=y_lim[0], top=y_lim[1])

        if x_label:
            plt.xlabel(x_label)

        if y_label:
            plt.ylabel(y_label)

        if results_dir:
            pass
            # plt.savefig(os.path.join(results_dir, title))

        return fig

    @staticmethod
    def plot_signals_mean_std(
        signal_fourplets,
        results_dir=None,
        title=None,
        legend=None,
        y_lim=None,
        x_label=None,
        y_label=None,
        confidence=0.68,
        alpha=0.5,
    ):

        factor = norm.ppf(1 / 2 + confidence / 2)  # 0.95 -> 1.959963984540054

        fig = plt.figure()
        for signal_fourplet in signal_fourplets:

            t = signal_fourplet[0]
            x = signal_fourplet[1]
            label = signal_fourplet[2]
            window = signal_fourplet[3]

            if window:
                x_mean = (
                    pd.Series(x)
                    .rolling(window, min_periods=1, center=True)
                    .mean()
                    .values
                )
                x_std = (
                    pd.Series(x)
                    .rolling(window, min_periods=1, center=True)
                    .std()
                    .values
                )

                plt.plot(t, x_mean, label=label)
                plt.fill_between(
                    t,
                    x_mean - factor * x_std,
                    x_mean + factor * x_std,
                    alpha=alpha,
                    label="{}_{} %".format(label, int(100 * confidence)),
                )
            else:
                plt.plot(t, x, label=label)

        if title:
            plt.title(title)

        if legend:
            plt.legend(loc=legend)
        else:
            plt.legend()

        if y_lim:
            plt.gca().set_ylim(bottom=y_lim[0], top=y_lim[1])

        if x_label:
            plt.xlabel(x_label)

        if y_label:
            plt.ylabel(y_label)

        if results_dir:
            pass
            # plt.savefig(os.path.join(results_dir, title))

        return fig
