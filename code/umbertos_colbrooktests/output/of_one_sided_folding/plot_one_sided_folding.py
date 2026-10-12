"""Three-panel comparison: only A*A, only AA*, both -- for three meshes."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

NU, EPS = 0.1, 0.5
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
ACCENT = "#eb6834"
# one hue, light -> dark; small sigma (deep inside the pseudospectrum) is dark
BLUES = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4",
         "#cde2fb", "#f3f8fe"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUES)

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": RULE, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.labelcolor": INK2, "text.color": INK,
    "mathtext.fontset": "cm", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

d = np.load(sys.argv[1])
X, Y, ref = d["X"], d["Y"], d["ref"]
levels = np.arange(-1.5, 0.51, 0.25)
eigs = [1 / (4 * NU) + NU * (n * np.pi) ** 2 for n in (1, 2, 3)]

COLS = [r"only $(A-z)^*(A-z)$:   $\sqrt{\mu_h^{(1)}(z)}$",
        r"only $(A-z)(A-z)^*$:   $\sqrt{\mu_h^{(2)}(z)}$",
        r"both:   $\Phi_h=\sqrt{\min\{\mu_h^{(1)},\mu_h^{(2)}\}}$"]
ROWS = [("u", "uniform", "uniform mesh, 16 cells"),
        ("g", "graded", r"mesh graded towards $x=0$, 16 cells"),
        ("r", "graded_x1", r"mesh graded towards $x=1$, 16 cells")]


def panel(ax, sig):
    cf = ax.contourf(X, Y, np.log10(sig), levels=levels, cmap=CMAP, extend="both")
    ax.contour(X, Y, ref, levels=[EPS], colors=INK, linewidths=1.0,
               linestyles=[(0, (4, 2.5))])
    ax.contour(X, Y, sig, levels=[EPS], colors=ACCENT, linewidths=1.6)
    ax.plot(eigs, [0] * 3, "o", ms=4.2, mfc=INK, mec="white", mew=0.9, zorder=5)
    ax.set_xlim(0, 15); ax.set_ylim(-10, 10)
    ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([-10, -5, 0, 5, 10])
    return cf


def mesh_rug(ax, nodes):
    """Small inset showing where the 17 mesh nodes sit in (0,1)."""
    ins = ax.inset_axes([0.05, 0.035, 0.52, 0.10])
    ins.set_facecolor("white")
    ins.vlines(nodes, 0, 1, color=INK, lw=0.6)
    ins.set_xlim(-0.02, 1.02); ins.set_ylim(0, 1)
    ins.set_yticks([]); ins.set_xticks([0, 1])
    ins.tick_params(length=0, pad=1.5, labelsize=6.5)
    ins.set_title(r"mesh on $\Omega=(0,1)$", fontsize=6.5, pad=2, color=INK2)
    for s in ins.spines.values():
        s.set_linewidth(0.4)


def figure(rows, out):
    # fixed margins in inches so every variant has the same panel size
    top_in, bottom_in, row_in = 0.70, 0.52, 2.69
    height = top_in + bottom_in + row_in * len(rows)
    fig, axes = plt.subplots(len(rows), 3, figsize=(9.6, height), squeeze=False,
                             sharex=True, sharey=True)
    for i, (tag, mesh, label) in enumerate(rows):
        s1, s2 = d[tag + "1"], d[tag + "2"]
        for j, sig in enumerate((s1, s2, np.minimum(s1, s2))):
            cf = panel(axes[i, j], sig)
            if i == 0:
                axes[i, j].set_title(COLS[j], fontsize=9, pad=6)
            if i == len(rows) - 1:
                axes[i, j].set_xlabel(r"$\mathrm{Re}\,z$")
        axes[i, 0].set_ylabel(label + "\n" + r"$\mathrm{Im}\,z$", fontsize=8.5)
        mesh_rug(axes[i, 0], d[mesh])

    fig.subplots_adjust(left=0.075, right=0.905, top=1 - top_in / height,
                        bottom=bottom_in / height,
                        wspace=0.07, hspace=0.09)
    pos0, pos1 = axes[0, -1].get_position(), axes[-1, -1].get_position()
    cax = fig.add_axes([0.92, pos1.y0, 0.012, pos0.y1 - pos1.y0])
    cb = fig.colorbar(cf, cax=cax, ticks=[-1.5, -1, -0.5, 0, 0.5])
    cb.outline.set_linewidth(0.4)
    cb.set_label(r"$\log_{10}$ of the plotted quantity", fontsize=8)
    handles = [
        Line2D([], [], color=ACCENT, lw=1.6,
               label=r"computed $\epsilon=0.5$ contour of the panel's quantity"),
        Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 2.5)),
               label=r"reference $\|(A-z)^{-1}\|^{-1}=0.5$"),
        Line2D([], [], color=INK, marker="o", ls="", ms=4.2, mec="white",
               mew=0.9, label=r"eigenvalues of $A$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=8, handlelength=2.6, columnspacing=2.2,
               bbox_to_anchor=(0.49, 1.0))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


figure(ROWS, "one_sided_folding")
figure(ROWS[:1], "one_sided_folding_uniform")
figure(ROWS[1:2], "one_sided_folding_graded")
figure(ROWS[2:], "one_sided_folding_graded_x1")
