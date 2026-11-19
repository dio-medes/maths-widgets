"""
Small implementation of geometric principle of Blackwell's Approachability theorem.

Makes a nice small gif in 2d where the initial point moves.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Plotting region [0, 10] x [0,10]
segment = np.linspace(0, 10, 500)

# Define the target set S
in_S = lambda x : (x[0] - 3) ** 2 + (x[1] - 3) ** 2 <= 2
X, Y  = np.meshgrid(segment, segment)
S = in_S((X, Y))

# A bounded sequence whose next element x_(n+1) always
# satisfies < x_(n+1) - pi_S(bar(x)_n) , bar(x)_n - pi_S(bar(x)_n) >

# These are some examples but there are infinitely many.
# Doing this programatically rather than exactly (i.e. with a projection
# formula for pi_S(x)).

def spin_sequence(init = [5, 5], N = 50, step = 3):
    xbar = [np.array(init)]
    seq = [np.array(init)]
    pts = np.column_stack([X[S], Y[S]])
    centre = pts.mean(axis=0)
    for n in range(1, N):
        p = xbar[-1]
        if in_S(p):
            walk_out = p
        else:
            proj = pts[np.argmin(((pts - p) ** 2).sum(axis=1))]
            dir = p - proj
            tangent_dir = np.array([dir[1], -dir[0]]) / np.linalg.norm(dir)
            walk_out = proj + (step / n) * tangent_dir
        lam = 0.1
        seq.append((1 - lam) * walk_out + lam * centre)
        xbar.append(xbar[-1] + (seq[-1] - xbar[-1]) / (n + 1))
    seq = np.stack(seq)
    xbar = np.stack(xbar)
    return seq, xbar

# Plotting

theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
r = 2.2 + 0.6 * np.cos(5 * theta)
inits = 3 + r[:, None] * np.column_stack([np.cos(theta), np.sin(theta)])

fig, ax = plt.subplots(figsize=(5, 5))
ax.contourf(X, Y, S, levels=[0.5, 1], colors="0.6", alpha=0.3)   # region S, drawn once
(seq_pts,) = ax.plot([], [], "o", ms=5, color="#2a78d6", label="Approaching sequence")
(xbar_line,) = ax.plot([], [], "o", lw=2, color="#eb6834", label="Average")
(start_pt,) = ax.plot([], [], "*", ms=12, color="#1baf7a", label="Initial point")
ax.set(xlim=(-2, 8), ylim=(-2, 8), aspect="equal")
ax.legend(loc="upper right")

def update(i):
    seq, xbar = spin_sequence(init = inits[i])
    seq_pts.set_data(seq[:, 0], seq[:, 1])
    xbar_line.set_data(xbar[:, 0], xbar[:, 1])
    start_pt.set_data([inits[i, 0]], [inits[i, 1]])
    return seq_pts, xbar_line, start_pt

ani = FuncAnimation(fig, update, frames=len(inits), blit=True)
ani.save("images/out.gif", writer=PillowWriter(fps=20), dpi=100)
