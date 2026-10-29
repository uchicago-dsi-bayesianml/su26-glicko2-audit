#!/usr/bin/env python3
"""Course-staff check: what does the notebook's all-players-at-once skill update target?

In every sampler of glicko_2_bayesian_modeling-3.ipynb (cells 8, 9 and 32), once the
latent performance differences D have been drawn, each player's skill is drawn from its
full conditional given the opponents' skills from the previous sweep, and all players
are drawn in the same step. That is a synchronous (Jacobi-style) update, not a Gibbs
sweep: a Gibbs sweep would condition each player on opponents already updated in the
same sweep. For a Gaussian conditional such as theta | D, the synchronous update has the
right mean at convergence but its own stationary covariance, which can differ from the
posterior's.

This script builds game graphs of 1,000 players, computes the exact posterior covariance
of the centred skills given D, and compares it with the stationary covariance of the
synchronous update, using the notebook's prior (sd 3) and its sum-to-zero centring. It
does this for uniformly random pairings, which is what the notebook's recovery checks
simulate (cells 11, 12 and 33), and for rating-matched pairings, where each game is
between players close in rank, as in Lichess's pools. It also measures how fast the mean
of the update approaches the posterior mean from a start at zero, the notebook's
initialisation for the static and two-skill models, and compares a sequential
(Gauss-Seidel) sweep, which targets the exact posterior.

The comparison holds D fixed, so it isolates the skill step. The full sampler also
redraws D every sweep, which slows mixing further; the mixing numbers here are an
optimistic bound.

Needs only NumPy, no data, and runs in a few seconds:
    python staff/check_synchronous_updates.py [--players 1000] [--games 250] [--seed 0]
"""
import argparse
import time

import numpy as np


def build_graph(rng, n, games_per_player, window):
    """Symmetric matrix of game counts. window=None: opponents uniform over the field;
    otherwise the opponent's rank is within +/- window of the player's rank."""
    n_games = n * games_per_player // 2
    w = rng.integers(0, n, n_games)
    if window is None:
        b = rng.integers(0, n, n_games)
    else:
        b = np.clip(w + rng.integers(-window, window + 1, n_games), 0, n - 1)
    keep = w != b
    w, b = w[keep], b[keep]
    A = np.zeros((n, n))
    np.add.at(A, (w, b), 1.0)
    np.add.at(A, (b, w), 1.0)
    return A


def stationary_cov(T, Q, tol=1e-13, max_doublings=60):
    """Solve S = T S T' + Q by doubling: S = sum_j T^j Q T'^j."""
    S, Tk = Q.copy(), T.copy()
    for _ in range(max_doublings):
        S = S + Tk @ S @ Tk.T
        Tk = Tk @ Tk
        if np.abs(Tk).max() < tol:
            return S
    raise RuntimeError("doubling did not converge")


def gap_recovered(M, m_star, contrast, checkpoints):
    """Fraction of the posterior-mean contrast reached after k sweeps from m = 0.

    The error e = m - m_star of the mean evolves as e <- M e, so after k sweeps the mean
    is m_star + e and the fraction reached is (contrast . m) / (contrast . m_star)."""
    e = -m_star.copy()
    target = contrast @ m_star
    out, k = {}, 0
    for stop in checkpoints:
        while k < stop:
            e = M @ e
            k += 1
        out[stop] = 1.0 + (contrast @ e) / target
    return out


def run(label, A, prior_sd, rng, checkpoints):
    n = A.shape[0]
    P = np.eye(n) - 1.0 / n                      # sum-to-zero centring, as in the notebook
    d = 1.0 / prior_sd**2 + A.sum(1)             # conditional precision of each player
    Lam = np.diag(d) - A                         # posterior precision of theta given D

    exact = P @ np.linalg.inv(Lam) @ P
    T = P @ (A / d[:, None]) @ P                 # synchronous update, then centring
    Q = P @ np.diag(1.0 / d) @ P
    sync = stationary_cov(T, Q)

    sd_ratio = np.sqrt(np.diag(sync) / np.diag(exact))
    c = np.zeros(n)                              # top tenth minus bottom tenth of the field
    c[: n // 10], c[-(n // 10):] = -10.0 / n, 10.0 / n
    contrast_ratio = np.sqrt(c @ sync @ c / (c @ exact @ c))
    rho = np.abs(np.linalg.eigvals(T)).max()

    # posterior mean for one simulated data set: true skills sorted by rank, Gaussian D
    theta = np.sort(rng.normal(0.0, 1.0, n))
    iu = np.triu_indices(n, 1)
    counts = A[iu].astype(int)
    i_idx, j_idx = np.repeat(iu[0], counts), np.repeat(iu[1], counts)
    D = theta[i_idx] - theta[j_idx] + rng.normal(0.0, 1.0, len(i_idx))
    rhs = np.zeros(n)
    np.add.at(rhs, i_idx, D)
    np.add.at(rhs, j_idx, -D)
    m_star = P @ np.linalg.solve(Lam, rhs)

    L = np.tril(A, -1)
    U = np.triu(A, 1)
    G = P @ np.linalg.solve(np.diag(d) - L, U)   # one sequential sweep in rank order, then centring
    rec_sync = gap_recovered(T, m_star, c, checkpoints)
    rec_seq = gap_recovered(G, m_star, c, checkpoints)

    print(f"\n{label}")
    print(f"  games per player: median {np.median(A.sum(1)):.0f}")
    print(f"  posterior sd, synchronous update / exact: median {np.median(sd_ratio):.3f}, "
          f"range {sd_ratio.min():.3f} to {sd_ratio.max():.3f}")
    print(f"  sd of (top tenth minus bottom tenth), synchronous / exact: {contrast_ratio:.3f}")
    print(f"  spectral radius of the synchronous update: {rho:.5f} "
          f"(about {1.0 / (1.0 - rho):,.0f} sweeps per e-fold of the slowest error)")
    print("  share of the posterior-mean gap between top and bottom tenth reached from zero:")
    print("    sweeps        " + "".join(f"{k:>8d}" for k in checkpoints))
    print("    synchronous   " + "".join(f"{rec_sync[k]:8.3f}" for k in checkpoints))
    print("    sequential    " + "".join(f"{rec_seq[k]:8.3f}" for k in checkpoints))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", type=int, default=1000)
    ap.add_argument("--games", type=int, default=250,
                    help="games per player; the notebook's training set averages 258 (cell 20)")
    ap.add_argument("--prior-sd", type=float, default=3.0, help="sigma_theta in cells 8 and 32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    checkpoints = [50, 250, 500, 2000]
    t0 = time.time()
    n = args.players
    print(f"{n} players, {args.games} games each, prior sd {args.prior_sd}, seed {args.seed}")
    for label, window in [
        ("uniform random pairing (as in the notebook's recovery checks)", None),
        (f"rating-matched, opponent within +/-{n // 5} ranks (nearest 40% of the field)", n // 5),
        (f"rating-matched, opponent within +/-{n // 20} ranks (nearest 10% of the field)", n // 20),
        (f"rating-matched, opponent within +/-{n // 50} ranks (nearest 4% of the field)", n // 50),
    ]:
        run(label, build_graph(rng, n, args.games, window), args.prior_sd, rng, checkpoints)
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
