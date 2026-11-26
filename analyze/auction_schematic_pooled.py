"""Worked example of one auction with POOLED lane bids (plan "pooled"), the companion of
auction_schematic.py: every car of a lane bids from its own balance, the lane's bid is the sum, the
contested winner's cars each pay their own bid, and the payment is shared among all cars of the
losing lanes. The numbers are chosen so that pooling changes the winner: N's front car outbids
E's (6 against 5), but E's longer queue bids more in total.

Writes analyze/figures/update/auction_schematic_pooled.pdf/.png and figures_schematic_pooled.tex.

    python analyze/auction_schematic_pooled.py
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from auction_schematic import CAR, GREEN, HALF, LINE, RED, ROAD  # noqa: E402
from fixed_rules_analysis import OUT, WIDTH, save  # noqa: E402
from parameter_analysis import INK, INK2  # noqa: E402

LIM = 7.6
GAP = 1.8   # distance between the centres of consecutive cars in a lane

# approach: (front car centre, heading unit vector, the cars' bids from the front back, label)
APPROACHES = {
    "N": ((-0.9, 2.95), (0, -1), (6, 4), "N: 6 + 4 = 10\neach car gets +3"),
    "S": ((0.9, -2.95), (0, 1), (2, 1), "S: 2 + 1 = 3\neach car gets +3"),
    "E": ((2.95, 0.9), (-1, 0), (5, 4, 3), "E: 5 + 4 + 3 = 12\npays 12 (contested),\neach car its own bid"),
    "W": ((-2.95, -0.9), (1, 0), (3, 1), "W: 3 + 1 = 4\npays 0\n(uncontested)"),
}
WINNERS = {"E", "W"}


def car(ax, centre, heading, bid: int) -> None:
    """A car 1.0 wide and 1.5 long pointing along `heading`, with its bid written on it."""
    w, l = (1.0, 1.5) if heading[0] == 0 else (1.5, 1.0)
    ax.add_patch(FancyBboxPatch((centre[0] - w / 2, centre[1] - l / 2), w, l,
                                boxstyle="round,pad=0,rounding_size=0.25", fc=CAR, ec=CAR,
                                lw=0.8, zorder=3))
    ax.text(centre[0], centre[1], str(bid), ha="center", va="center", fontsize=7.5,
            color="white", fontweight="bold", zorder=4)


def intersection(ax) -> None:
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
    for k, ((x, y), (dx, dy), bids, _) in APPROACHES.items():
        for n, bid in enumerate(bids):
            car(ax, (x - n * GAP * dx, y - n * GAP * dy), (dx, dy), bid)
        if k in WINNERS:   # its movement through the box
            start = (x + 0.85 * dx, y + 0.85 * dy)
            end = (x + 6.0 * dx, y + 6.0 * dy)
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>,head_length=4,head_width=2.4",
                                         color=GREEN, lw=1.6, zorder=2.5))
    label = {"N": ((-HALF - 0.35, 3.85), "right", "center"),
             "S": ((HALF + 0.35, -3.85), "left", "center"),
             "E": ((4.45, HALF + 0.35), "center", "bottom"),
             "W": ((-3.85, -HALF - 0.35), "center", "top")}
    for k, ((x, y), ha, va) in label.items():
        ax.text(x, y, APPROACHES[k][3], ha=ha, va=va, fontsize=7.5, color=INK, linespacing=1.3)
    ax.text(-HALF - 0.35, 6.6, "number on a car:\nits own bid", ha="right", va="center",
            fontsize=6.5, color=INK2, linespacing=1.2)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.axis("off")


STEPS = [
    ("1  Bids", "Every car within 50 m of the stop line bids from its own balance by its own "
                "rule, and a lane's bid is the sum: E 12, N 10, W 4, S 3."),
    ("2  Winners", "Highest bid first; skip any bid that conflicts with one already accepted."),
    (None, [("E 12", "accepted"), ("N 10", "skipped: conflicts with E"),
            ("W 4", "accepted: compatible with E"), ("S 3", "skipped: conflicts with E")]),
    ("3  Payments", "E is contested: without its bid, N (10) would have won, so E pays 12, each "
                    "car its own bid (5, 4 and 3). W is uncontested, so it pays 0."),
    ("4  Rebates", "The 12 paid is split equally between the four cars of the losing lanes, "
                   "N and S: +3 each. Total carma is unchanged."),
    ("5  Grant", "E and W turn green until every car on those lanes at the award has crossed; "
                 "N and S wait at red."),
    ("Why pooling matters", "With only front cars bidding, N would have won (6 beats 5). "
                            "Summed, E's longer queue outbids N."),
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
    text = r"""% Worked example of one auction with pooled lane bids (analyze/auction_schematic_pooled.py).
\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth]{auction_schematic_pooled.pdf}
    \caption{The auction of Figure~\ref{fig:auction_schematic} with pooled lane bids: every car
    within 50\,m bids from its own balance by its own rule (number on each car) and a lane's bid
    is the sum. The greedy rule (eq.~\eqref{eq:greedy}) accepts E, then W; E is contested and
    pays its total, each car its own bid; W is uncontested and pays nothing; and the 12 paid is
    shared equally among all four cars of the losing lanes. With only front cars bidding, N would
    have won (6 beats 5); pooled, the longer queue on E outbids it. Stop lines show the signal
    after the auction: green for E and W, red for N and S.}
    \label{fig:auction_schematic_pooled}
\end{figure}
"""
    with open(os.path.join(OUT, "figures_schematic_pooled.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    height = 3.75
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0.0, 0.02, 0.46, 0.96])
    intersection(ax)
    steps(fig, 0.47, 0.975, 0.125 / height)
    save(fig, "auction_schematic_pooled")
    latex()
