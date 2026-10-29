<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Beyond Glicko-2: Hierarchical Bayesian Estimation of Latent Chess Skill

> **Example project from Summer 2026.** This is a copy of [jholiphant/Auditing-Glicko-2](https://github.com/jholiphant/Auditing-Glicko-2) at commit 078328a, without its commit history, because an earlier version of the notebook listed real account names. John Henry Oliphant's own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project) and a short script, `staff/check_synchronous_updates.py`, that demonstrates one of its points. The games come from the Lichess open database, which is released under CC0, and are not in this repository; [data/README.md](data/README.md) explains how the author built the table.

Lichess rates its players with Glicko-2, which keeps a Gaussian belief about each player's skill and updates it after every game: an approximate Bayesian filter, with shortcuts that let it run in real time. Opponents' ratings are treated as known, the rate at which skill drifts is set by hand, and the only thing it learns from a game is who won. What does the full Bayesian model say, and what is the information Glicko-2 ignores worth? John Henry Oliphant streamed a year of the Lichess database into a table of 5.2 million games between 21,488 very active blitz players, wrote three samplers from scratch in JAX (one skill per player, a monthly skill trajectory with a learned drift, and separate board and clock skills), and compared each with the Lichess ratings on a two-month holdout.

## Summary

**Data.** Twelve monthly archives of the Lichess open database, June 2025 to May 2026: the games between 21,603 players with at least 5,000 blitz games, with the ten training months' blitz games sampled at 20 percent, the two holdout months kept in full, and every game of the 114 accounts with Lichess's BOT title removed. That leaves 5,199,271 games, 2,773,946 for training and 2,425,325 in the holdout, each with the outcome, month, time control, whether it ended on the clock, and both players' Lichess ratings before the game.

**Model.** An ordered probit: a latent performance difference, the skill gap plus a white advantage h plus unit noise, decides a White win, a draw within ±γ, or a Black win. The static model gives each player one skill; the dynamic model gives a skill per month, linked by a random walk with step size τ and started at the player's Lichess rating; the two-skill model gives a board strength β and a clock discipline κ, with whether a game ends on time and who wins on the clock modelled through κ.

**Inference.** Hand-written samplers in JAX: truncated-normal data augmentation, conjugate Gaussian skill updates (forward filtering and backward sampling for the monthly trajectories) and Metropolis steps for the thresholds, each first checked on simulated data. Each real fit is one chain of 2,000 sweeps, about two to seven minutes on a Colab GPU (cells 24, 27 and 34).

**Baseline and evaluation.** A "ladder" of ordered-probit predictors built on the Lichess rating difference, adding a scale, a draw margin and a white advantage in turn, each fit by maximum likelihood on the training games. All models are scored on the holdout months by the ranked probability score (RPS), lower being better.

**Part A.** Neither Bayesian model beats the fully tuned ladder: 0.48471 for the static model and 0.48486 for the dynamic one, against 0.48351. The write-up draws two findings from it: the Lichess ratings run about 25 percent "hot" (the fitted scale is 0.437 where 0.588 would be neutral), and skill drifts far less than Glicko-2 assumes (τ ≈ 0.03 per month).

**Part B.** Board strength and clock discipline come out mildly anti-correlated (−0.285), κ correlates −0.94 with players' flag rates and −0.04 with their ratings, and on 1.66 million holdout games decided on the board β scores 0.46077 against 0.48047 for a re-tuned Glicko baseline: the write-up's headline, a better board rating than Glicko-2.

The staff section below explains why the Part B comparison uses information that is only known after each game, why Part A is not the same-data comparison it is described as, what the single chains do and do not show, what the samplers' simultaneous update of all players does to the posterior spread, and which parts of the title and the plan were not built.

## Team and contributions

Team "The Master Bayesians", Summer 2026, a team of one.

- **John Henry Oliphant** ([@jholiphant](https://github.com/jholiphant)): everything, in 23 commits: the streaming pipeline and the five scripts it calls (which stayed in the author's Google Drive), the pipeline notebook, the modelling notebook with its three samplers and their simulation checks, the bot removal through the Lichess API, the baseline ladder, Parts A and B, the plate diagram, and the write-up.

The repository ran from August 12 to August 17, 2026. Every commit is an upload through GitHub's web interface, and the notebooks were run in Colab, so the history records what was uploaded and when, not how the work was written. The attributions follow the author's own [commit history](https://github.com/jholiphant/Auditing-Glicko-2/commits/main), since this copy has none.

## Getting started

Neither the games nor the pipeline scripts are in the repository; [data/README.md](data/README.md) explains how the table was built. The notebooks were written for Colab with a GPU and Python 3.11, and there is no requirements file. They need JAX, NumPy, SciPy, DuckDB and requests, and the pipeline adds zstandard and PyArrow.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install "jax[cuda12]" numpy scipy duckdb requests zstandard pyarrow jupyter
```

Cell numbers in this README count every cell of a notebook, markdown included, from 0 at the top.

**The pipeline.**

1. `chess_data_pipeline-3.ipynb` runs the four stages (discovery, cohort, enrichment, assembly) by calling the author's scripts, then checks the assembled table and builds a first baseline table. The two streaming passes over twelve months take hours each.
2. `glicko_2_bayesian_modeling-3.ipynb`, top to bottom: the samplers (cells 7 to 9), the simulation checks (11, 12), bot removal (14 to 18), the clean table (20), the baseline ladder (22), the static model (24), the dynamic model (26, 27), the Part A comparison (29), and Part B (32 to 38).

**Staff check.** `staff/check_synchronous_updates.py` builds game graphs of 1,000 players, with opponents drawn at random or matched by rank, and compares the exact posterior of the skills with what the notebook's skill update converges to, and how fast. It needs only NumPy and runs in a few seconds:

```sh
python staff/check_synchronous_updates.py
```

**What does not run as committed.** Both notebooks mount Google Drive with `google.colab` and read from the author's folder. The pipeline notebook imports `lichess_stream` and calls four more scripts, none of them in the repository; its outputs are cleared, and its cell 23 writes hard-coded ladder scores to `baseline_ladder.json` instead of the computed ones. The modelling notebook needs `model_data.npz`, the monthly Parquet files (cell 15 reads them to map ids to usernames) and the title cache `titles.json`; without the cache, cell 16 queries the Lichess API for 21,603 accounts.

## Data

- **Source:** the [Lichess open database](https://database.lichess.org/), one compressed PGN file per month, each about 30 GB compressed and holding 87 to 95 million games in the months the recorded run shows.
- **Terms:** CC0, a public-domain dedication. The games carry public usernames.
- **The table:** 16.4 million games between the 21,603 cohort players, 5.55 million after the 20 percent sample of training blitz games (both from the recorded pipeline run that survives in an earlier version of the pipeline notebook; see [data/README.md](data/README.md)), and 5,199,271 between 21,488 players, in one connected graph, after removing 114 BOT-titled accounts and their 347,673 games (cells 14, 17, 18 and 20 of the modelling notebook). The draw rate is 4.6 percent (cell 20) and 28 percent of training games end on the clock (cell 34).
- **Formats:** in the recorded run, 93 percent of the games are blitz, with 260,279 bullet, 108,936 rapid and 21,024 classical games. The modelling notebook pools them and never reads the time-control column.
- **Not in the repository:** the scripts, the Parquet files, the assembled table (122 MB) and the title cache.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The author's write-up: how Glicko-2 works, the model, the sampler, Part A, Part B, a note on production use, references. |
| `chess_data_pipeline-3.ipynb` | The data pipeline, its checks and a first baseline ladder. Calls five scripts that are not in the repository; outputs cleared. |
| `glicko_2_bayesian_modeling-3.ipynb` | Everything else: the samplers and their simulation checks, bot removal, the baseline ladder, Part A and Part B. |
| `data/README.md` | Where the games come from, their terms, and how the table was built. |
| `staff/check_synchronous_updates.py` | Course staff's check of the samplers' skill update, described above. |
| `.github/assets/` | The program banner. |

## Method

**Likelihood.** For a game between White *w* and Black *b*, D = θ*w* − θ*b* + h + ε with ε ~ Normal(0, 1); the outcome is a White win if D > γ, a draw if −γ ≤ D ≤ γ, and a Black win if D < −γ. The skill scale is fixed by the unit noise.

**Static model (cell 8).** θ*i* ~ Normal(0, 3²). Each sweep draws every D from its truncated normal by the inverse CDF, then every θ from its Gaussian conditional given D and the opponents' skills from the previous sweep, centres the skills, and updates h and γ by random-walk Metropolis on the likelihood with D integrated out, which is valid because D is redrawn at the start of the next sweep.

**Dynamic model (cell 9).** Skill per player and month, θ*i,t* = θ*i,t−1* + Normal(0, τ²), with θ*i,0* ~ Normal(seed*i*, 1 + τ²) and τ² ~ InverseGamma(2, 0.001). The seed is the player's earliest Lichess rating in the training window, centred, divided by 173.7 and multiplied by the ladder's fitted scale (cell 26). Given D, each player's twelve-month trajectory is drawn jointly by forward filtering and backward sampling, τ² has a conjugate inverse-gamma update, and h and γ are updated as before.

**Two-skill model (cell 32).** β*i*, κ*i* ~ Normal(0, 3²). Every game has a flag latent with mean −(κ*w* + κ*b*) + c, positive if the game ended on time; decisive games decided on the board have a latent with mean β*w* − β*b* + h, truncated at ±γ, and decisive games decided on time a race latent with mean κ*w* − κ*b*. β is updated from the board latents, κ from the flag and race latents, and h, γ and c by Metropolis. Board draws enter the likelihood for h and γ but not the update of β.

**Baselines (cell 22).** With Δ the Lichess rating difference divided by 173.7, each rung predicts an ordered probit in *s*Δ + h with draw margin γ: B0 fixes *s* = 1 and γ = h = 0, B0b fits *s*, B1 adds γ and B2 adds h. The rungs are fit by maximum likelihood on the training games. B-null predicts the training outcome frequencies.

**Evaluation.** Predictions are for April and May 2026. The static model uses its training posterior and the dynamic model the skill at the last training month (cell 29), both averaged over 750 posterior draws (2,000 sweeps, 500 discarded, every second kept); the two-skill predictions in cells 36 to 38 use posterior means. The score is the ranked probability score over Black win, draw and White win.

### What is Bayesian here

A great deal, all written by hand: the Albert and Chib augmentation that makes an ordinal likelihood conjugate, forward filtering and backward sampling of whole trajectories, a conjugate update for the drift variance, Metropolis-within-Gibbs for the thresholds, posterior predictive averaging for the Part A forecasts, and simulation checks of every sampler, two with interval coverage. The bot removal starts from a posterior: 17 of the 20 highest-skill accounts in the first fit carry the BOT title (cell 17).

What stops short: the title's hierarchy is one shared drift variance; the player priors have fixed spreads and the cross-format covariance Σ in the plate diagram was not built. The skill update moves all players at once rather than sweeping through them, which approximates the posterior rather than sampling it (see below). Each fit is one chain without convergence diagnostics, and no player's posterior interval is reported or checked on real games, so the uncertainty that motivates the project is never used. The Glicko ladder is fit by maximum likelihood, which is fine for a baseline.

## Results

**Simulation checks.** From the modelling notebook.

| Check | Setting | Result |
|---|---|---|
| Static (cell 11) | 300 players, about 40,000 random pairings | skill correlation 0.995, 95% interval coverage 0.963, γ 0.147 (true 0.15), h 0.059 (true 0.05) |
| Dynamic (cell 12) | 300 players, 12 months, about 48,000 random pairings, prior seeded with the true starting skills | τ 0.149 ± 0.007 (true 0.15), skill correlation 0.988, coverage 0.943, correlation of change over the year 0.836 (the cell's own target is 0.9) |
| Two-skill (cell 33) | 400 players, about 120,000 random pairings, independent β and κ | correlation with truth 0.993 (β) and 0.998 (κ), recovered corr(β, κ) +0.075, γ, h and c within 0.002 of the truth |

**The baseline ladder (cell 22).** Holdout RPS, fitted on the training games.

| Rung | Adds | Holdout RPS |
|---|---|---:|
| B-null | training outcome frequencies | 0.49835 |
| B0 | Lichess rating difference, probit slope 1 | 0.49514 |
| B0b | fitted scale | 0.48517 |
| B1 | draw margin | 0.48411 |
| B2 | white advantage (*s* = 0.4372, γ = 0.0597, h = 0.0446) | 0.48351 |

**Part A (cells 24, 27 and 29).** Holdout RPS on the same games.

| Holdout games | Count | Dynamic, seeded | Static | Glicko B2 |
|---|---:|---:|---:|---:|
| All | 2,425,325 | 0.48486 | 0.48471 | 0.48351 |
| `returning` stratum | 34,689 | 0.48816 | 0.48774 | 0.48489 |
| `large_gap` stratum | 62,495 | 0.24019 | 0.24061 | 0.24432 |

The strata are defined in the missing `assemble.py`. The dynamic fit's τ averages 0.0330 with a standard deviation of 0.0019 over the kept draws.

**Part B, the fitted model (cells 34 and 35).** corr(β, κ) = −0.285 over players; c = −0.679, γ = 0.072, h = 0.058. Over players with at least 50 training games, κ correlates −0.936 with the player's flag rate in the training games, β correlates +0.878 with the player's first Lichess rating in the window, and κ correlates −0.040 with it.

**Part B, prediction.** Holdout RPS on decisive games only; draws, 4.6 percent of games, are left out of every Part B score. The three columns of the β/κ model differ in how a game's predictor is chosen.

| Holdout games | Count | Glicko B2 | β/κ, predictor chosen by the game's own termination (cell 36) | β/κ, termination marginalized (cell 37) |
|---|---:|---:|---:|---:|
| All decisive | 2,314,298 | 0.48434 | 0.44480 | 0.48840 |
| Decided on the board | 1,662,256 | 0.48190 | 0.46077 | 0.47363 |
| Decided on the clock | 652,042 | 0.49058 | 0.40407 | 0.52607 |

Cell 36 is the version the author identified as leaking, since whether a game ends on time is known only after it; cell 37, the author's correction, uses the model's own probability that the game ends on time. Cell 38 scores the board channel alone on the 1,662,256 board-decided games (0.46077, the same predictor and number as cell 36's middle row) against the B2 ladder (0.48190) and the ladder refitted to those games (0.48047). The write-up's Finding 4 table reports these and calls them the corrected result. **Correction:** the corrected, marginalized model scores 0.47363 on those games and loses on all decisive games, the one Part B comparison that does not select games by how they ended, 0.48840 against 0.48434.

**Staff check of the skill update.** From `staff/check_synchronous_updates.py`: 1,000 players with about 250 games each (the training set averages 258, cell 20) and the notebook's prior. The last column is the share of the posterior-mean gap between the top and bottom tenth of players reached after 500 sweeps from zero, the static and two-skill fits' start and burn-in.

| Pairing | Median player sd, update / exact | Top-minus-bottom-tenth sd, update / exact | Sweeps per e-fold of the slowest error | Gap reached, simultaneous / sequential |
|---|---:|---:|---:|---:|
| Uniformly random | 1.001 | 1.000 | 1 | 1.000 / 1.000 |
| Opponent in the nearest 40% of the field | 0.997 | 0.762 | 15 | 1.000 / 1.000 |
| Nearest 10% | 0.945 | 0.710 | 215 | 0.926 / 0.993 |
| Nearest 4% | 0.846 | 0.708 | 861 | 0.621 / 0.791 |

## What to learn from this project

Course staff added this section. The project is technically ambitious and most of its machinery is correct; the points below are about what its comparisons can and cannot support.

- **Score only with what is known before the game.** Part B's headline, β beating "even a board-tuned Glicko-2" by 0.0197 RPS, is computed in cell 38 on holdout games selected by their own outcome: decisive, and not ended on time. Both are known only after the game. The board channel is by construction the right predictor for games already known to be decided on the board, so its advantage there says that β summarizes board results better than a rating that also absorbs clock results, not that it forecasts better. Its 0.46077 comes from the same predictor as the leaking row of cell 36 that the author caught and replaced, and the "board-tuned" baseline is fitted on those same holdout games (`nll_board` uses `y_ho[bd]`), which favours the baseline but is still tuning on the test set. The prospective comparison is cell 37, where the two-skill model loses, 0.48840 against 0.48434 on all decisive games, and no Part B model is scored on all games with draws. Fix the evaluation set before looking at outcomes, fit every baseline parameter on training data, and make the prospective number the headline.
- **If you call a comparison information-fair, make it so, or say what differs.** Part A is described as using "the same data as Glicko-2". The Bayesian models see only games inside the cohort, with the training months' blitz games sampled at 20 percent (`--subsample-frac 0.20` in cell 14 of the pipeline notebook, hence 2,773,946 training games for ten months against 2,425,325 holdout games for two, cell 20), and they are frozen at March 2026 for the whole holdout (cell 29). The Lichess rating on each holdout game is the rating just before it, updated by every earlier game, including earlier holdout games and games against players outside the cohort. Coming within 0.0012 RPS of that online filter from a fifth of the cohort's games is respectable, not evidence that proper inference loses on equal terms. Update the Bayesian model month by month through the holdout, or freeze the Lichess ratings at the same cut-off.
- **Run more than one chain, and look at it.** Every real fit is one chain with seed 0, traced only by a line printed every 250 sweeps. γ, h and c are flat from sweep 250. τ is not: 0.0420 at sweep 250, 0.0370 at 500 where burn-in ends, then 0.0355, 0.0338, 0.0322, 0.0312 and 0.0313 (cell 27), so the reported 0.0330 ± 0.0019 averages a drifting chain and its spread describes the drift. Several chains, R-hat, effective sample sizes and a trace plot would settle it. The comparison with Glicko-2 is also not made in common units: 0.033 per month on the probit scale is about 0.033 × 173.7 / 0.437 ≈ 13 rating points, while Glicko-2's τ limits how fast a player's volatility changes and is not a drift rate.
- **Check that your sampler samples the posterior.** In cells 8, 9 and 32 every player's skill is drawn given the opponents' skills from the previous sweep, all players at once. That is not a Gibbs sweep; for a Gaussian conditional it converges to the right mean but a different covariance. With opponents drawn uniformly at random, as in all three simulation checks, the staff script finds it exact to within 0.1 percent, so the checks could not see the problem. With opponents matched by rank, as on Lichess, it understates the posterior standard deviation of the gap between the strongest and weakest tenth of players by 24 to 29 percent and individual players' by up to 25 percent, and its slowest errors take hundreds of sweeps to fade. Slow mixing comes with single-player updates on such a graph (after 500 sweeps on the tightest graph a sequential sweep reaches 79 percent of the top-to-bottom gap, the simultaneous one 62 percent); the understated spread comes from the shortcut. Point estimates look unharmed: the zero-started static model and the Lichess-seeded dynamic model score within 0.00015 of each other (cell 29). The size of the effect on the real cohort depends on how tightly Lichess pairs players, which the repository cannot show. Updating players in independent groups (a graph colouring), or drawing all skills jointly from the sparse Gaussian, is exact at similar cost.
- **Validate a latent variable on data it was not fit to.** κ's correlation of −0.936 with players' flag rates is computed on the training flags κ was fit to (cell 35), so it shows that the model fits its data, not that κ is a skill. κ mostly measures how often a player's games end on time, since every game gives a flag observation and only decisive games on the clock give a race observation, and a player who often wins on the clock is pulled in opposite directions by the two channels. Its correlation of −0.04 with the rating says that flag frequency is unrelated to strength, not that Glicko-2 is "blind to clock discipline"; a rating counts a win on time like any other. Check κ against holdout flag rates.
- **Be careful when calling a rating miscalibrated.** The fitted scale of 0.437 against 0.588 is a useful observation: in this cohort, rating gaps predict less than Glicko-2's formula implies. Part of that is expected of any noisy estimate, since noise in a predictor flattens its fitted slope, and the value moves with the cohort: the recorded pipeline run, bots included, fitted 0.528 (see data/README.md). The ladder also overstates the gain from rescaling: its "raw Glicko" rung uses a probit slope of 1 rather than the 0.588 that approximates Glicko-2's logistic formula, so the 0.00997 gain from B0 to B0b partly converts between link functions.
- **Build what the title promises, or change the title.** Cell 1's plate diagram and the pipeline notebook ("the cross-format covariance Σ is a headline result") describe format-specific skills sharing a covariance across bullet, blitz, rapid and classical. The modelling notebook never reads the time-control column, so the hierarchy is one shared drift variance. The value-of-information analysis promised in cell 31, bits per game from termination, game length and clock use, was not run.
- **Think before naming real people.** An earlier version of the notebook had a Part C, a heavy-tailed random walk whose variance inflation scored accounts for sandbagging, account sharing or engine use, and printed the top accounts by username. The author removed it on August 14, but the write-up still says the model "occasionally catches a cheater" and lists "all three parts", and the final notebook prints three accounts without a BOT title as bot suspects. Course staff redacted those names and left the earlier version out. An anomaly score is not evidence about a person: report such results without names and leave accusations to Lichess's fair-play process.
- **Ship the code that made the data, and show the reader a picture.** The five scripts that build the table were never committed, so the cohort, the sample and the strata can be checked only against an earlier version's printed outputs. Neither notebook draws a single plot: no trace plot, skill trajectory, β against κ or calibration curve. Twelve of the modelling notebook's fifteen markdown cells lost their line breaks, so their headings run into the text, and cell 0 still gives the numbers from before the bots were removed (21,603 players, a baseline of 0.47988). Comments such as "These are the numbers your model has to beat. Save them." are drafting notes that should have been edited out.

What the project does well: a pipeline that streams a year of the Lichess database, about a billion games, through memory and keeps a connected graph of 5.2 million; three samplers written from scratch, with augmentation, forward filtering and backward sampling, conjugate updates and Metropolis steps, correct apart from the simultaneous update; a simulation check for every sampler before it saw real games, two with interval coverage; bot removal that starts from the posterior and is confirmed against the Lichess API; a baseline ladder fit on training games and scored prospectively; an honest negative result in Part A; a leak in cell 36 that the author found and fixed in cell 37, with a caveat that the clock channel does not help prospectively; and a clear write-up of a question that deserves the fairer comparisons described above.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Summer 2026. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff copied the author's repository and made these edits. The author's code is unchanged; the only change to the author's files is the redaction below.

- Imported the author's three files as they were at commit 078328a of [jholiphant/Auditing-Glicko-2](https://github.com/jholiphant/Auditing-Glicko-2), as one commit without the earlier history, because an earlier version of the modelling notebook listed real account names.
- In the saved outputs of cells 15 and 17 of `glicko_2_bayesian_modeling-3.ipynb`, replaced the names of three accounts without a BOT title, which the notebook lists as bot suspects, with `[account 1]` to `[account 3]`, padded to the same width. Nothing else in the notebook changed.
- Added a `.gitignore` that keeps Lichess archives, Parquet extracts, the assembled tables and the title cache out of the repository. There was no clutter to remove.
- Added `data/README.md`, on where the games come from and how the table was built, and `staff/check_synchronous_updates.py`, the check described above.
- Renamed the author's README to `REPORT.md`, unchanged, and added this README with the program banner.

## License

The author has not added a license, so the work is shown here under GitHub's terms for public repositories. Ask the author before reusing it. The Lichess database is CC0, as above.
