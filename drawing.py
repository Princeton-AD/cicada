import hls4ml
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import numpy.typing as npt

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from typing import List, Callable


class Draw:
    def __init__(self, output_dir: Path = Path("plots")):
        self.output_dir = output_dir
        self.cmap = ["green", "red", "blue", "orange", "purple", "brown"]
        self.signals = ["Zero Bias", "SUEP", "HtoLongLived", "VBHFto2C", "TT", "SUSYGGBBH"]
        self.models_long = ["cicada", "section", "super"]
        self.models_short = ["cic", "scn", "spr"]
        self.signals_cmap = {}
        for key, value in zip(self.signals, self.cmap):
            self.signals_cmap[key] = value
        self.models_cmap = {}
        for key, value in zip(self.models_long, self.cmap):
            self.models_cmap[key] = value
        for key, value in zip(self.models_short, self.cmap):
            self.models_cmap[key] = value
        hep.style.use("CMS")

    def _parse_name(self, name: str) -> str:
        return name.replace(" ", "-").lower()

    def _save_fig(self, name: str) -> None:
        plt.savefig(
            f"{self.output_dir}/{self._parse_name(name)}.png", bbox_inches="tight"
        )
        if self.interactive:
            plt.show()
        plt.close()

    def plot_loss_history(
        self, training_loss: npt.NDArray, validation_loss: npt.NDArray, name: str, ylim = [1.0, 3.0]
    ):
        plt.plot(np.arange(1, len(training_loss) + 1), training_loss, label="Training")
        plt.plot(np.arange(1, len(validation_loss) + 1), validation_loss, label="Validation")
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(ylim[0], ylim[1])
        plt.savefig(
            f"{self.output_dir}/{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_multiple_loss_history(
        self, losses, name: str, ylim = [1.0, 3.0]
    ):
        nLosses = len(losses)
        for i in range(nLosses):
            plt.plot(np.arange(1, len(losses[i][0]) + 1), losses[i][0], label=f"Training, {losses[i][2]}", color=self.cmap[i])
            plt.plot(np.arange(1, len(losses[i][1]) + 1), losses[i][1], label=f"Validation, {losses[i][2]}", color=self.cmap[i], linestyle="dotted")
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(ylim[0], ylim[1])
        plt.savefig(
            f"{self.output_dir}/{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_loss_histories(
        self, loss_dict: dict[str, (npt.NDArray, npt.NDArray)], name: str
    ):
        for model_name, (train_loss, val_loss) in loss_dict.items():
            c = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(np.arange(1, len(train_loss) + 1), train_loss, color=c, label=f"{model_name} (Training)")
            plt.plot(np.arange(1, len(val_loss) + 1), val_loss, color=c, ls=":", label=f"{model_name} (Validation)")
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        self._save_fig(name)

    def plot_deposits(self, deposits: npt.NDArray, name: str):
        plt.imshow(deposits, vmin=0, vmax = deposits.max(), cmap="Purples")
        plt.xlabel(r"i$\eta$")
        plt.ylabel(r"i$\phi$")
        plt.savefig(
            f"{self.output_dir}/deposits_{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_regional_deposits(self, deposits: npt.NDArray, mean: float, name: str):
        im = plt.imshow(
            deposits.reshape(6, 14), vmin=0, vmax=deposits.max(), cmap="Purples"
        )
        ax = plt.gca()
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(r"Calorimeter E$_T$ deposit (GeV)")
        plt.xticks(np.arange(14), labels=np.arange(4, 18))
        plt.yticks(
            np.arange(6),
            labels=np.arange(6)[::-1],
            rotation=90,
            va="center",
        )
        plt.xlabel(r"i$\eta$")
        plt.ylabel(r"i$\phi$")
        plt.title(rf"Mean E$_T$ {mean: .2f} ({name})")
        plt.savefig(
            f"{self.output_dir}/profiling_mean_deposits_{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_spacial_deposits_distribution(
        self, deposits: List[npt.NDArray], labels: List[str], name: str
    ):
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        for deposit, label in zip(deposits, labels):
            bins = np.argwhere(deposit)
            phi, eta = bins[:, 1], bins[:, 2]
            ax1.hist(
                eta + 4,
                density=True,
                facecolor=None,
                bins=np.arange(4, 19),
                label=label,
                histtype="step",
            )
            ax2.hist(
                phi,
                density=True,
                facecolor=None,
                bins=np.arange(19),
                label=label,
                histtype="step",
            )
        ax1.set_ylabel("a.u.")
        ax1.set_xlabel(r"i$\eta$")
        ax2.set_xlabel(r"i$\phi$")
        plt.legend(loc="best")
        plt.savefig(
            f"{self.output_dir}/profiling_spacial_{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_deposits_distribution(
        self, deposits: List[npt.NDArray], labels: List[str], name: str
    ):
        for deposit, label in zip(deposits, labels):
            plt.hist(
                deposit.reshape((-1)),
                bins=100,
                range=(0, 1024),
                density=1,
                label=label,
                log=True,
                histtype="step",
            )
        plt.xlabel(r"E$_T$")
        plt.legend(loc="best")
        plt.savefig(
            f"{self.output_dir}/profiling_deposits_{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_reconstruction_results(
        self,
        deposits_in: npt.NDArray,
        deposits_out: npt.NDArray,
        loss: float,
        name: str,
    ):
        fig, (ax1, ax2, ax3, cax) = plt.subplots(
            ncols=4, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}
        )
        max_deposit = deposits_in.max()

        ax1 = plt.subplot(1, 4, 1)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title("Original", fontsize=18)
        ax1.imshow(
            deposits_in.reshape(18, 14), vmin=0, vmax=max_deposit, cmap="Purples"
        )

        ax2 = plt.subplot(1, 4, 2)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title("Reconstructed", fontsize=18)
        ax2.imshow(
            deposits_out.reshape(18, 14), vmin=0, vmax=max_deposit, cmap="Purples"
        )

        ax3 = plt.subplot(1, 4, 3)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(rf"|$\Delta$|, MSE: {loss: .5f}", fontsize=18)

        im = ax3.imshow(
            np.abs(deposits_in - deposits_out).reshape(18, 14),
            vmin=0,
            vmax=max_deposit,
            cmap="Purples",
        )

        ip = InsetPosition(ax3, [1.05, 0, 0.05, 1])
        cax.set_axes_locator(ip)
        fig.colorbar(im, cax=cax, ax=[ax1, ax2, ax3]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )

        plt.savefig(
            f"{self.output_dir}/reconstruction_results_{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_reconstruction_results_scn(
        self,
        deposits_in: npt.NDArray,
        deposits_out: npt.NDArray,
        loss: float,
        name: str,
    ):
        fig, (ax1, ax2, ax3, cax) = plt.subplots(
            ncols=4, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}
        )
        max_deposit = max(deposits_in.max(), deposits_out.max())

        ax1 = plt.subplot(1, 4, 1)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title("Original", fontsize=18)
        ax1.imshow(
            deposits_in.reshape(6, 14), vmin=0, vmax=max_deposit, cmap="Purples"
        )

        ax2 = plt.subplot(1, 4, 2)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title("Reconstructed", fontsize=18)
        ax2.imshow(
            deposits_out.reshape(6, 14), vmin=0, vmax=max_deposit, cmap="Purples"
        )

        ax3 = plt.subplot(1, 4, 3)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(rf"|$\Delta$|, MSE: {loss: .2f}", fontsize=18)

        im = ax3.imshow(
            np.abs(deposits_in - deposits_out).reshape(6, 14),
            vmin=0,
            vmax=max_deposit,
            cmap="Purples",
        )

        ip = InsetPosition(ax3, [1.05, 0, 0.05, 1])
        cax.set_axes_locator(ip)
        fig.colorbar(im, cax=cax, ax=[ax1, ax2, ax3]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )

        plt.savefig(
            f"{self.output_dir}/reconstruction_results_{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_phi_shift_variance(
        self, losses: List[float], name: str
    ):
        x = np.arange(len(losses))
        loss_means = np.mean(losses, axis=1)
        plt.plot(x, loss_means)
        loss_stds =  np.std(losses, axis=1)
        lower = loss_means - loss_stds / 2
        upper = loss_means + loss_stds / 2
        plt.fill_between(x, lower, upper, alpha=0.1)
        plt.xlabel(r"Shift [$\Delta$ i$\phi$]")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel(r"$\Delta_{rel} (MSE)$")
        plt.axvline(x=0, color='grey', linestyle=':', label='Original')
        plt.axvline(x=18, color='grey', linestyle=':')
        plt.axhline(y=loss_means[0], color='grey', linestyle=':')
        plt.legend()
        self._save_fig(name)

    def plot_anomaly_score_distribution(
        self, scores: List[npt.NDArray], labels: List[str], name: str, xlim = [0, 256]
    ):
        for score, label in zip(scores, labels):
            if label in self.signals_cmap: color = self.signals_cmap[label]
            elif label in self.models_cmap: color = self.models_cmap[label]
            score_tmp = np.array([])
            for i in range(score.shape[0]):
                score_tmp = np.append(score_tmp, score[i].flatten())
            plt.hist(
                score_tmp.reshape((-1)),
                bins=100,
                range=(xlim[0], xlim[1]),
                density=1,
                label=label,
                log=True,
                histtype="step",
                color=color,
            )
        plt.xlabel(r"Anomaly Score")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"{self.output_dir}/score_dist_{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_roc_curve(
        self,
        y_trues: List[npt.NDArray],
        y_preds: List[npt.NDArray],
        labels: List[str],
        inputs: List[npt.NDArray],
        name: str,
        cv: int = 3,
    ):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        auc_return = []
        for y_true, y_pred, label, d in zip(
            y_trues, y_preds, labels, inputs
        ):
            aucs = []
            for _, indices in skf.split(y_pred, y_true):
                fpr, tpr, _ = roc_curve(y_true[indices], y_pred[indices])
                aucs.append(auc(fpr, tpr))
            std_auc = np.std(aucs)

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            auc_return.append([roc_auc, std_auc])
            fpr_base, tpr_base, _ = roc_curve(y_true, np.mean(d**2, axis=(1, 2)))
            if label in self.signals_cmap: color = self.signals_cmap[label]
            elif label in self.models_cmap: color = self.models_cmap[label]
            plt.plot(
                fpr * 28.61,
                tpr,
                linestyle="-",
                lw=1.5,
                color=color,
                alpha=0.8,
                label=rf"{label} (AUC = {roc_auc: .4f} $\pm$ {std_auc: .4f})",
            )

            plt.plot(
                fpr_base * 28.61,
                tpr_base,
                linestyle="--",
                lw=1.0,
                color=color,
                alpha=0.5,
                label=rf"{label}, Baseline",
            )

        plt.plot(
            [0.003, 0.003],
            [0, 1],
            linestyle="--",
            lw=1,
            color="black",
            label="3 kHz",
        )
        plt.xlim([0.0002861, 28.61])
        plt.ylim([0.01, 1.0])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Trigger Rate (MHz)")
        plt.ylabel("Signal Efficiency")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"{self.output_dir}/roc_{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()
        return np.array(auc_return)

    def plot_compilation_error(
        self, scores_keras: npt.NDArray, scores_hls4ml: npt.NDArray, name: str
    ):
        plt.scatter(scores_keras, np.abs(scores_keras - scores_hls4ml), s=1)
        plt.xlabel("Anomaly Score, $S$")
        plt.ylabel("Error, $|S_{Keras} - S_{hls4ml}|$")
        plt.savefig(
            f"{self.output_dir}/compilation_error_{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_compilation_error_distribution(
        self, scores_keras: npt.NDArray, scores_hls4ml: npt.NDArray, name: str
    ):
        plt.hist(scores_keras - scores_hls4ml, fc="none", histtype="step", bins=100)
        plt.xlabel("Error, $S_{Keras} - S_{hls4ml}$")
        plt.ylabel("Number of samples")
        plt.yscale("log")
        plt.savefig(
            f"{self.output_dir}/compilation_error_dist_{self._parse_name(name)}.png",
            bbox_inches="tight",
        )

    def plot_cpp_model(self, hls_model, name: str):
        hls4ml.utils.plot_model(
            hls_model,
            show_shapes=True,
            show_precision=True,
            to_file=f"{self.output_dir}/cpp_model_{self._parse_name(name)}.png",
        )

    def plot_roc_curve_comparison(
        self, scores_keras: dict, scores_hls4ml: npt.NDArray, name: str
    ):
        fpr_model: list = []
        tpr_model: list = []

        scores_keras_normal = scores_keras["Background"]
        scores_hls4ml_normal = scores_hls4ml["Background"]

        for dataset_name, color in zip(list(scores_keras.keys())[:-1], self.cmap):
            scores_keras_anomaly = scores_keras[dataset_name]
            scores_hls4ml_anomaly = scores_hls4ml[dataset_name]

            y_true = np.append(
                np.zeros(len(scores_keras_normal)), np.ones(len(scores_hls4ml_anomaly))
            )
            y_score_keras = np.append(scores_keras_normal, scores_keras_anomaly)
            y_score_hls = np.append(scores_hls4ml_normal, scores_hls4ml_anomaly)

            for y_scores, model, ls in zip(
                [y_score_keras, y_score_hls], ["Keras", "hls4ml"], ["-", "--"]
            ):
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                plt.plot(
                    fpr * 28.61,
                    tpr,
                    linestyle=ls,
                    color=color,
                    label="{0}: {1}, AUC = {2:.4f}".format(
                        model, dataset_name, auc(fpr, tpr)
                    ),
                )

        plt.plot(
            [0.003, 0.003],
            [0, 1],
            linestyle="--",
            color="black",
            label="3 kHz trigger rate",
        )
        plt.xlim([0.0002861, 28.61])
        plt.ylim([0.01, 1.0])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Trigger Rate (MHz)")
        plt.ylabel("Signal Efficiency")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"{self.output_dir}/compilation_roc_{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_output_reference(self):
        with open("misc/output-reference.txt") as f:
            data = f.read()
        data = np.array([row.split(",") for row in data.split("\n")[:-1]]).astype(
            np.int8
        )
        data = np.flipud(data) - 1
        legend_elements = [
            Patch(
                facecolor=self.cmap[0],
                edgecolor=self.cmap[0],
                label="Anomaly Detection, Integer Part",
            ),
            Patch(
                facecolor=self.cmap[1],
                edgecolor=self.cmap[1],
                label="Anomaly Detection, Decimal Part",
            ),
            Patch(
                facecolor=self.cmap[2], edgecolor=self.cmap[2], label="Heavy Ion Bit"
            ),
            Patch(facecolor=self.cmap[3], edgecolor=self.cmap[3], label="Reserved"),
        ]
        plt.figure(figsize=(25, 5))
        plt.pcolor(
            data, edgecolors="black", alpha=0.6, cmap=ListedColormap(self.cmap[:4])
        )
        plt.xticks([])
        plt.yticks([])
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    abs(y * 32 + x - 191),
                    horizontalalignment="center",
                    fontsize=16,
                    verticalalignment="center",
                )
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(0, 0),
            loc="upper left",
            frameon=False,
            ncol=4,
            borderaxespad=0,
        )
        plt.savefig(
            f"{self.output_dir}/ugt_link_reference.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_results_supervised(
        self, grid: npt.NDArray, models: list[str], datasets: list[str], name: str
    ):
        plt.imshow(grid, alpha=0.7, cmap="RdYlGn")
        plt.xticks(
            np.arange(len(models)),
            labels=models,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        plt.yticks(np.arange(len(datasets)), labels=datasets)
        for i in range(len(datasets)):
            for j in range(len(models)):
                text = plt.text(
                    j,
                    i,
                    "{0:.3f}".format(grid[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    size=16,
                )
        plt.savefig(
            f"{self.output_dir}/supervised_{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_scatter_score_comparison(
        self, x: npt.NDArray, y: npt.NDArray, x_title: str, y_title: str, name: str, limits: str = "fit"
    ):
        for signal in list(self.signals_cmap.keys()):
            if signal in name:
                scatter_color = self.signals_cmap[signal]
                label = signal
        x_tmp = np.array([])
        y_tmp = np.array([])
        for i in range(len(x)):
            x_tmp = np.append(x_tmp, x[i].flatten())
            y_tmp = np.append(y_tmp, y[i].flatten())
        x_tmp = np.reshape(x_tmp, (-1))
        y_tmp = np.reshape(y_tmp, (-1))
        plt.scatter(x_tmp, y_tmp, s=1, color = scatter_color, label = label)
        if(limits=="equal"):
            max_val_x = np.sort(np.array([x_tmp, y_tmp]), axis=None)[int(-0.05 * (x_tmp.shape[0]+y_tmp.shape[0]))]
            max_val_y = max_val_x
        elif(limits=="fit"):
            max_val_x = np.sort(np.array([x_tmp]), axis=None)[int(-0.05 * x_tmp.shape[0])]
            max_val_y = np.sort(np.array([y_tmp]), axis=None)[int(-0.05 * y_tmp.shape[0])]
        elif(limits=="equalsignal"):
            max_val_x = 16
            max_val_y = 16
        elif(limits=="fitmax"):
            max_val_x = np.max(np.array([x_tmp]))
            max_val_y = np.max(np.array([y_tmp]))
        elif(limits=="fitmaxequal"):
            max_val_x = np.max(np.array([x_tmp, y_tmp]))
            max_val_y = max_val_x
        elif(limits=="equalstudent"):
            max_val_x = 27
            max_val_y = 27
        plt.xlim(0, max_val_x)
        plt.ylim(0, max_val_y)
        def f(x, m):
            return m*x
        popt, pcov = curve_fit(f, x_tmp, y_tmp)
        x_line = np.linspace(0, max_val_x)
        y_line = f(x_line, popt[0])
        plt.plot(x_line, y_line, color = "black", label = f"m = {popt[0]:.2f}")
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.legend()
        plt.savefig(
            f"{self.output_dir}/scatter_score_{self._parse_name(name)}.png",
            bbox_inches="tight",
        )
        plt.close()
        return popt

    def plot_score_comparison_distributions(
        self, x: npt.NDArray, x_title: str, y_title: str, name: str, limits: str = "equalsignal"
    ):
        if(limits=="equalsignal"):
            xmax = 5
            xmin = -5
        #score_dist = [[], []]
        diff = [[], []]
        stats = [[], []]
        for i in range(len(self.signals)):
            for j in range(len(self.models_long)-1):
                diff[j].append(np.array(x[j+1][i]) - np.array(x[0][i]))
                stats[j].append(norm.fit(diff[j][-1]))
                #score_dist[j].append([np.std(diff[j][-1]), np.mean(diff[j][-1])])
        for i in range(len(self.signals)):
            for j in range(len(self.models_long)-1):
                color = self.models_cmap[self.models_long[j+1]]
                label = f"{self.models_long[j+1]}-{self.models_long[0]}: mu = {stats[j][i][0]:.2f}, s = {stats[j][i][1]:.2f}"
                #label = f"{self.models_long[j+1]}-{self.models_long[0]}: mu = {score_dist[j][i][1]:.2f}, s = {score_dist[j][i][0]:.2f}"
                plt.hist(diff[j][i], alpha=0.5, bins=100, range=(xmin,xmax), density=1, histtype="step", color=color, label=label)
            plt.ylim(0, 1.0)
            plt.xlabel(x_title)
            plt.ylabel(y_title)
            plt.legend(loc="upper left")
            plt.savefig(
                f"{self.output_dir}/score_comparison_dist_{self.signals[i]}_{self._parse_name(name)}.png",
                bbox_inches="tight",
            )
            plt.close()
        for i in range(len(self.models_long)-1):
            for j in range(len(self.signals)):
                color = self.signals_cmap[self.signals[j]]
                label = f"{self.signals[j]}: mu = {stats[i][j][0]:.2f}, s = {stats[i][j][1]:.2f}"
                #label = f"{self.signals[j]}: mu = {score_dist[i][j][1]:.2f}, s = {score_dist[i][j][0]:.2f}"
                plt.hist(diff[i][j], alpha=0.5, bins=100, range=(xmin,xmax), density=1, histtype="step", color=color, label=label)
            plt.ylim(0, 1.0)
            plt.xlabel(x_title)
            plt.ylabel(y_title)
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.savefig(
                f"{self.output_dir}/score_comparison_dist_{self.models_short[i+1]}_{self.models_short[0]}_{self._parse_name(name)}.png",
                bbox_inches="tight",
            )
            plt.close()
        return stats

    def plot_anomaly_scores_distribution(
        self, score_list: List[List[npt.NDArray]], label_list: List[str], name: str
    ):
        for scores, label in zip(score_list, label_list):
            for score in scores:
                score_tmp = np.array([])
                for i in range(len(score)):
                    score_tmp = np.append(score_tmp, score[i].flatten())
                plt.hist(
                    score_tmp.reshape((-1)),
                    alpha=0.5,
                    bins=100,
                    range=(0, 256),
                    density=1,
                    label=label,
                    log=True,
                    histtype="step",
                    color=self.signals_cmap[label]
                )
        plt.xlabel("Anomaly Score")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(
            f"{self.output_dir}/scores_dist_{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_mean_sectioned_deposits(
        self, deposits: npt.NDArray, name: str
    ):
        # Assumes deposits.shape is (-1, 3, 6, 14, 1)
        deposits = np.mean(deposits, axis=0)
        deposits = np.reshape(deposits, (18, 14))
        plt.imshow(
            deposits, vmin=0, vmax=np.max(deposits), cmap="Purples"
        )
        plt.title("Mean ET")
        plt.xlabel("Eta")
        plt.ylabel("Phi")
        plt.savefig(
            f"{self.output_dir}/mean_deposits_scn_{self._parse_name(name)}.png", bbox_inches="tight"
        )
        plt.close()

    def plot_mse(
        self, mse: npt.NDArray, bottleneck_sizes: npt.NDArray,
    ):
        # mse has shape (model type, signal type, bottleneck size, 2)
        for i in range(mse.shape[1]):
            for j in range(mse.shape[0]):
                plt.plot(bottleneck_sizes[0], mse[j,i,:,0], color = self.models_cmap[self.models_long[j]], label = self.models_long[j])
                #plt.errorbar(bottleneck_sizes[0], mse[j,i,:,0], yerr = mse[j,i,:,1], color = self.models_cmap[self.models_long[j]], label = self.models_long[j])
            plt.title(f"MSE ({self.signals[i]}) vs Latent Space")
            plt.xlabel("Latent Space")
            plt.ylabel("MSE")
            plt.legend()
            plt.savefig(
                f"{self.output_dir}/mse_{self._parse_name(self.signals[i])}.png", bbox_inches="tight"
            )
            plt.close()
        for i in range(mse.shape[0]):
            for j in range(mse.shape[1]):
                plt.plot(bottleneck_sizes[0], mse[i,j,:,0], color = self.signals_cmap[self.signals[j]], label = self.signals[j])
                #plt.errorbar(bottleneck_sizes[0], mse[i,j,:,0], yerr = mse[i,j,:,1], color = self.signals_cmap[self.signals[j]], label = self.signals[j])
            plt.title(f"MSE ({self.models_long[i]}) vs Latent Space")
            plt.xlabel("Latent Space")
            plt.ylabel("MSE")
            plt.legend()
            plt.savefig(
                f"{self.output_dir}/mse_{self._parse_name(self.models_short[i])}.png", bbox_inches="tight"
            )
            plt.close()

    def plot_auc(
        self, auc: npt.NDArray, bottleneck_sizes: npt.NDArray,
    ):
        # auc has shape (signal type, model type, bottleneck size, 2)
        for i in range(auc.shape[1]):
            for j in range(auc.shape[0]):
                plt.errorbar(bottleneck_sizes[0], auc[j,i,:,0], yerr = auc[j,i,:,1], color = self.models_cmap[self.models_long[j]], label = self.models_long[j])
            plt.title(f"AUC ({self.signals[i]}) vs Latent Space")
            plt.xlabel("Latent Space")
            plt.ylabel("AUC")
            plt.legend()
            plt.savefig(
                f"{self.output_dir}/AUC_{self._parse_name(self.signals[i+1])}.png", bbox_inches="tight"
            )
            plt.close()
        for i in range(auc.shape[0]):
            for j in range(auc.shape[1]):
                plt.errorbar(bottleneck_sizes[0], auc[i,j,:,0], yerr = auc[i,j,:,1] , color = self.signals_cmap[self.signals[j+1]], label = self.signals[j+1])
            plt.title(f"AUC ({self.models_long[i]}) vs Latent Space")
            plt.xlabel("Latent Space")
            plt.ylabel("AUC")
            plt.legend()
            plt.savefig(
                f"{self.output_dir}/AUC_{self._parse_name(self.models_short[i])}.png", bbox_inches="tight"
            )
            plt.close()

    def make_equivariance_plot(
        self,
        image: npt.NDArray,
        f: Callable[npt.NDArray, npt.NDArray],  # symmetry transformation
        g: Callable[npt.NDArray, npt.NDArray],  # mapping of the model
        name: str
    ):

        fig, axs = plt.subplots(
            nrows=2, ncols=4, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}
        )
        max_deposit = image.max()
        xmax, ymax, _ = image.shape

        mse_g_1 = float(np.mean((g(image) - image)**2))
        mse_gf_f = float(np.mean((g(f(image)) - f(image))**2))
        mse_gf_fg = float(np.mean((g(f(image)) - f(g(image)))**2))

        axs[0, 0].imshow(image, vmin=0, vmax=max_deposit, cmap="Purples")
        axs[0, 1].imshow(f(image), vmin=0, vmax=max_deposit, cmap="Purples")
        im = axs[0, 2].imshow(g(f(image)), vmin=0, vmax=max_deposit, cmap="Purples")
        ip = InsetPosition(axs[0][2], [1.05, 0, 0.05, 1])
        axs[0][3].set_axes_locator(ip)
        fig.colorbar(im, cax=axs[0][3], ax=axs[0][:-1]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )

        axs[1, 0].imshow(image, vmin=0, vmax=max_deposit, cmap="Purples")
        axs[1, 1].imshow(g(image), vmin=0, vmax=max_deposit, cmap="Purples")
        im = axs[1, 2].imshow(f(g(image)), vmin=0, vmax=max_deposit, cmap="Purples")
        ip = InsetPosition(axs[1][2], [1.05, 0, 0.05, 1])
        axs[1][3].set_axes_locator(ip)
        fig.colorbar(im, cax=axs[1][3], ax=axs[1][:-1]).set_label(
            label=r"Calorimeter E$_T$ deposit (GeV)", fontsize=18
        )

        axs[0, 0].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[0, 0].text(xmax-3.5, ymax/2+1, 'trans', fontsize=18)

        axs[0, 1].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[0, 1].text(xmax-3.5, ymax/2+1, 'pred', fontsize=18)
        axs[0, 1].text(xmax-4, ymax/2+3, rf"MSE: {mse_gf_f:.1f}", fontsize=16)

        axs[1, 0].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[1, 0].text(xmax-3.5, ymax/2+1, 'pred', fontsize=18)
        axs[1, 0].text(xmax-4, ymax/2+3, rf"MSE: {mse_g_1:.1f}", fontsize=16)

        axs[1, 1].annotate('', xy=(1.4, 0.5), xycoords='axes fraction', 
                           xytext=(1.0, 0.5), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='->'))
        axs[1, 1].text(xmax-3.5, ymax/2+1, 'trans', fontsize=18)

        axs[0, 2].annotate('', xy=(0.5, -0.2), xycoords='axes fraction', 
                           xytext=(0.5, 0), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', arrowstyle='<->'))
        axs[0, 2].text(xmax/2-1.5, ymax+6, rf"MSE: {mse_gf_fg:.2f}", fontsize=16)

        for row in axs:
            for ax in row[:-1]:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        self._save_fig(name)
