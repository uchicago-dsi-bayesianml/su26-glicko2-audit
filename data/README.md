# Data

No game data is in this repository. The author built the modelling table from the Lichess open database with five Python scripts that are not in the repository either, and kept the results in a Google Drive folder (`/content/drive/MyDrive/Colab Notebooks/Bayes ML Final Project/data`). Both notebooks mount that folder in their first code cell. This note records what the table contains, where it came from, and what it would take to rebuild it.

## Where the data comes from

The [Lichess open database](https://database.lichess.org/) publishes every rated game played on lichess.org as one compressed PGN file per month, with headers (players, result, ratings before the game, time control, termination) and, for most games, the clock time after every move.

**Terms.** "Database exports are released under the Creative Commons CC0 license." CC0 is a public-domain dedication, so no attribution is required; citing the database URL and the months used is still good practice. The games include Lichess usernames, which are public, but an analysis that singles out real accounts by name should be careful about what it implies about them (see the staff README).

**Size.** Each monthly standard-chess file is roughly 30 GB compressed and 200 GB decompressed, by the pipeline notebook's own description. In the months that the author's recorded run shows, each file held 87 to 95 million games.

## How the author built the table

`chess_data_pipeline-3.ipynb` lists the four stages and calls a script for each. The scripts (`lichess_stream.py`, `pass1_discovery.py`, `cohort.py`, `pass2_enrich.py` and `assemble.py`) sat in the author's Drive folder next to the notebooks and were never committed, so their logic is known only from the command lines, the notebook's markdown and the outputs of an earlier version of the notebook (`chess_data_pipeline-2.ipynb`, in the author's own repository's history; the version here has its outputs cleared). The numbers below come from that recorded run.

1. **Stream.** Twelve monthly archives, June 2025 to May 2026, are streamed over HTTP and decompressed in memory, so nothing large touches the disk. Headers are parsed for every game; movetext, with its `[%clk]` comments, only in the second pass.
2. **Discovery.** Per-player activity counts in blitz across the twelve months: 3,891,003 distinct players in the window.
3. **Cohort.** Players with at least 5,000 blitz games in the window and at least six active months (`--min-games 5000 --min-active-months 6`): 21,603 players, with an average of 7,223 games each and an average of 11.6 active months.
4. **Enrichment.** The corpus is streamed again and a game is kept only if both players are in the cohort, in any time control. A month took about 50 minutes and kept about 1.4 to 1.5 million games, saved as a Parquet file of about 45 MB. All twelve months gave 16,412,774 games.
5. **Assembly** (`--holdout-months 2 --bucket month --subsample-frac 0.20 --return-pct 99 --return-min-days 3`). Monthly time buckets 0 to 11; buckets 10 and 11 (April and May 2026) are the holdout and are kept in full; of the training buckets, non-blitz games (339,975) are kept in full and 20 percent of the blitz games are kept at random (2,716,458 of 13,582,288). The result is `model_data.npz`, 5,546,944 games (3,056,433 training, 2,490,511 holdout), 122 MB, one connected comparison graph. The assembly step also writes stratum flags (`stratum_returning`, `stratum_large_gap` and others) whose exact definitions are in the missing `assemble.py`.

## Removing bots

Cells 13 to 18 of `glicko_2_bayesian_modeling-3.ipynb` fit the static model to the assembled table, look up every one of the 21,603 usernames with the public Lichess users endpoint (`POST https://lichess.org/api/users`, in batches of 200 with a pause and a back-off on HTTP 429, cached in `titles.json`), and drop every game involving an account with the BOT title: 114 accounts and 347,673 games (6.27 percent). Accounts without the title are kept, including the three high-skill accounts that the notebook lists as suspects; their names are redacted in this copy. The cleaned table, `model_data_nobots.npz`, has 5,199,271 games between 21,488 players, 2,773,946 in training and 2,425,325 in the holdout (cell 20), and is still one connected graph.

## What the table holds

Per game, as the write-up's schema table lists: `white` and `black` (player ids), `y` (0 black win, 1 draw, 2 white win), `t_idx` (month 0 to 11), `c_idx` (bullet, blitz, rapid, classical), `flagged` (the game ended on time), `is_train`, and `glicko_white` and `glicko_black`, the players' Lichess ratings before the game, read from the PGN headers. The table also carries the stratum flags and `n_plies`, and the enriched Parquet files carry per-side clock features from the movetext. The modelling notebook uses none of these except two stratum flags in cell 29, and never uses `c_idx`. The blitz cohort plays mostly blitz with each other: in the recorded run, 5,156,705 of the 5,546,944 games were blitz, 260,279 bullet, 108,936 rapid and 21,024 classical.

## Rebuilding it

Without the five scripts the table cannot be rebuilt exactly. The pipeline notebook's markdown and arguments describe each stage well enough to reimplement it, and the stages are standard: a streaming `zstandard` reader over the monthly `.pgn.zst` files, header parsing, DuckDB for the counts, and NumPy for the integer arrays. Expect the two passes over twelve months to take a day or more of streaming. The Lichess API lookup for titles needs no token but is rate-limited. The `.gitignore` in this repository keeps any rebuilt files out of version control.
