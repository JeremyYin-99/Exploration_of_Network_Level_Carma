"""Worked example of one carma auction (formulation.tex, eqs. 4-7) for the update document:
four approaches bid, the greedy rule picks the compatible winners, the contested winner pays and
the losers share the payment, then the grant.

Writes analyze/figures/update/auction_schematic.pdf/.png and figures_schematic.tex.

    python analyze/auction_schematic.py
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from fixed_rules_analysis import OUT, WIDTH, save  # noqa: E402
from parameter_analysis import INK, INK2  # noqa: E402

ROAD, LINE = "#e6e5df", "#ffffff"
GREEN, RED = "#2b8a57", "#c4453a"
CAR, FOLLOW = "#52514e", "#bdbcb3"
HALF = 1.8   # half road width
LIM = 7.2

# approach: (front car centre, heading unit vector, bid, outcome lines)
APPROACHES = {
    "N": ((-0.9, 3.05), (0, -1), 12, "pays 12\n(contested)"),
    "S": ((0.9, -3.05), (0, 1), 7, "pays 0\n(uncontested)"),
    "E": ((3.05, 0.9), (-1, 0), 9, "gets +6"),
    "W": ((-3.05, -0.9), (1, 0), 3, "gets +6"),
}
WINNERS = {"N", "S"}


def car(ax, centre, heading, color, edge):
    """A car 1.0 wide and 1.7 long, centred at `centre` and pointing along `heading`."""
    w, l = (1.0, 1.7) if heading[0] == 0 else (1.7, 1.0)
    ax.add_patch(FancyBboxPatch((centre[0] - w / 2, centre[1] - l / 2), w, l,
                                boxstyle="round,pad=0,rounding_size=0.25", fc=color, ec=edge,
                                lw=0.8, zorder=3))


def intersection(ax):
    ax.add_patch(Rectangle((-HALF, -LIM), 2 * HALF, 2 * LIM, fc=ROAD, ec="none", zorder=0))
    ax.add_patch(Rectangle((-LIM, -HALF), 2 * LIM, 2 * HALF, fc=ROAD, ec="none", zorder=0))
    for a, b in ((HALF, LIM), (-LIM, -HALF)):   # centre lines outside the box
        ax.plot([0, 0], [a, b], color=LINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.plot([a, b], [0, 0], color=LINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    # stop lines, coloured by the signal each approach gets after the auction
    stops = {"N": ([-HALF, 0], [HALF, HALF]), "S": ([0, HALF], [-HALF, -HALF]),
             "E": ([HALF, HALF], [0, HALF]), "W": ([-HALF, -HALF], [-HALF, 0])}
    for k, (xs, ys) in stops.items():
        ax.plot(xs, ys, color=GREEN if k in WINNERS else RED, lw=3.2, solid_capstyle="butt",
                zorder=2)
    for k, ((x, y), (dx, dy), bid, outcome) in APPROACHES.items():
        car(ax, (x, y), (dx, dy), CAR, CAR)
        car(ax, (x - 2.1 * dx, y - 2.1 * dy), (dx, dy), "white", FOLLOW)   # rest of the lane
        if k in WINNERS:   # its movement through the box
            start = (x + 0.95 * dx, y + 0.95 * dy)
            end = (x + 6.0 * dx, y + 6.0 * dy)
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>,head_length=4,head_width=2.4",
                                         color=GREEN, lw=1.6, zorder=2.5))
    label = {"N": ((-HALF - 0.35, 3.05), "right", "center"),
             "S": ((HALF + 0.35, -3.05), "left", "center"),
             "E": ((3.6, HALF + 0.35), "center", "bottom"),
             "W": ((-3.6, -HALF - 0.35), "center", "top")}
    for k, ((x, y), ha, va) in label.items():
        _, _, bid, outcome = APPROACHES[k]
        ax.text(x, y, f"{k}: bids {bid}\n{outcome}", ha=ha, va=va, fontsize=7.5, color=INK,
                linespacing=1.3, fontweight="normal")
    ax.text(-HALF - 0.35, 5.15, "rest of lane:\nno bid", ha="right", va="center", fontsize=6.5,
            color=INK2, linespacing=1.2)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.axis("off")


STEPS = [
    ("1  Bids", "The front car of each lane within 50 m of the stop line bids for its whole "
                "lane: N 12, E 9, S 7, W 3."),
    ("2  Winners", "Highest bid first; skip any bid that conflicts with one already accepted."),
    (None, [("N 12", "accepted"), ("E 9", "skipped: conflicts with N"),
            ("S 7", "accepted: compatible with N"), ("W 3", "skipped: conflicts with N")]),
    ("3  Payments", "N is contested: without its bid, E (9) would have won, so N pays its own "
                    "bid, 12. S is uncontested: without it no conflicting bid wins, so S pays 0."),
    ("4  Rebates", "The 12 paid is split equally between the losing front cars: E +6, W +6. "
                   "Total carma is unchanged."),
    ("5  Grant", "N and S turn green until every car on those lanes at the award has crossed; E "
                 "and W wait at red. Then the intersection holds its next auction."),
]


def steps(fig, x0: float, y0: float, line: float, wrap: int = 54) -> None:
    """The step list as figure text, top-down from (x0, y0); `line` is one line in figure units."""
    y = y0
    for head, body in STEPS:
        if head is None:   # the greedy run, one row per bid
            for bid, what in body:
                fig.text(x0 + 0.03, y, bid, fontsize=7.5, color=INK, va="top",
                         fontweight="bold")
                fig.text(x0 + 0.095, y, what, fontsize=7.5, color=INK, va="top")
                y -= line
            y -= 0.5 * line
            continue
        fig.text(x0, y, head, fontsize=8, color=INK, va="top", fontweight="bold")
        y -= 1.05 * line
        for row in textwrap.wrap(body, wrap):
            fig.text(x0 + 0.03, y, row, fontsize=7.5, color=INK, va="top")
            y -= line
        y -= 0.45 * line


def latex() -> None:
    text = r"""% Worked example of one auction (analyze/auction_schematic.py).
\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth]{auction_schematic.pdf}
    \caption{One auction at a four-way intersection, assuming each lane only goes straight, so
    north (N) and south (S) are compatible and both conflict with east (E) and west (W). Only the
    front car of each lane bids (eq.~\eqref{eq:bid}); the greedy rule accepts N, then S
    (eq.~\eqref{eq:greedy}); N pays its bid because it displaced E, S rides free
    (eq.~\eqref{eq:payment}); and the losers share the payment (eq.~\eqref{eq:rebate}). Stop
    lines show the signal after the auction: green for N and S, red for E and W.}
    \label{fig:auction_schematic}
\end{figure}
"""
    with open(os.path.join(OUT, "figures_schematic.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    height = 3.3
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0.0, 0.02, 0.46, 0.96])
    intersection(ax)
    steps(fig, 0.47, 0.97, 0.125 / height)
    save(fig, "auction_schematic")
    latex()
