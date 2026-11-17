# Threshold CKKS for Privacy-Preserving Multi-Site Health Analytics

CYB 599 Graduate Project, Part 3: implementation and evaluation.
Author: Sathish Reddy Pidugu (sp3754@nau.edu)

Several hospitals compute pooled statistics and a linear regression over their
combined patient records, and no single party can decrypt anything on its own.
Each site holds one share of an n-out-of-n CKKS secret key. An untrusted server
aggregates ciphertexts and runs encrypted gradient descent, refreshing
ciphertexts with an interactive multiparty bootstrapping protocol. An analyst
receives only the final result, after every site contributes a noise-flooded
partial decryption. The protocol follows the Part 2 design document, and the
deviations found during implementation are listed below.

All data are **synthetic** (`src/tckks/data.py`); no real patient data is used.

## Quick start

Tested on **Linux x86_64, Ubuntu 24.04, Python 3.12** (2 vCPU, 7 GB RAM).

```bash
git clone <this-repo-url> cyb599-threshold-ckks
cd cyb599-threshold-ckks
./run.sh --quick     # tests + small evaluation, ~1-2 minutes
./run.sh             # full evaluation used in the report, ~10-15 minutes, ~4 GB RAM peak
```

`run.sh` creates `.venv/`, installs the pinned dependencies, runs the test
suite, runs the experiments, and writes the figures and a summary table:

```
results/results.json     raw measurements (every number in the report)
results/summary.md       tables generated from results.json
results/figures/*.pdf    figures used in the report
```

### Platform notes

OpenFHE's Python package (`openfhe` on PyPI) ships a compiled extension for
Linux x86_64 only:

| Environment | Works? | How |
|---|---|---|
| Ubuntu 24.04 / Debian 13, Python 3.12 | yes (tested) | `./run.sh` |
| Ubuntu 22.04, Python 3.10 | expected (wheel `1.6.1.0.22.4`) | `./run.sh` |
| Windows | via WSL2 (Ubuntu 24.04) | `./run.sh` inside WSL |
| macOS (Intel or Apple Silicon) | via Docker | see below |
| Linux without Python 3.12 but with [uv](https://docs.astral.sh/uv/) | yes | `run.sh` uses `uv venv --python 3.12` |

Docker (any OS; provided for convenience but not tested by the author, since the
test machine had no Docker daemon; emulated `linux/amd64` on Apple Silicon is several times slower):

```bash
docker build -t tckks .                                    # add --platform linux/amd64 on Apple Silicon
docker run --rm -v "$PWD/results:/app/results" tckks --quick
```

## What the code does

The table maps each phase of the Part 2 protocol to the code that implements it.

| Part 2 phase | Code | OpenFHE calls |
|---|---|---|
| 0. Setup: public bounds, parameters | `params.py`, `data.py` (`LOWER`, `UPPER`) | `CCParamsCKKSRNS`, `GenCryptoContext` |
| 1. Collective key generation | `ThresholdCKKS.keygen` | `MultipartyKeyGen`, `MultiKeySwitchGen`, `MultiAddEvalKeys`, `MultiMultEvalKey`, `MultiAddEvalMultKeys`, `MultiEvalAtIndexKeyGen`, `MultiAddEvalAutomorphismKeys` |
| 2. Local encryption (diagonal packing) | `run_mode_a`, `encrypt_regression_inputs` | `MakeCKKSPackedPlaintext`, `Encrypt` |
| 3. Server evaluation (Algorithm 1) | `run_regression`, `_rotations` | `EvalAdd`, `EvalMultNoRelin`, `Relinearize`, `Rescale`, `EvalFastRotation` |
| 4. Collective refresh | `refresh` | `IntMPBootAdjustScale`, `IntMPBootRandomElementGen`, `IntMPBootDecrypt`, `IntMPBootAdd`, `IntMPBootEncrypt` |
| 5. Threshold decryption + release policy | `decrypt`, `round_sig` | `MultipartyDecryptLead/Main/Fusion` |
| Non-private baseline | `baseline.py` | numpy only |

The protocol runs in two rounds:

1. **Round 1 (Mode A):** each site encrypts 17 moments (count, sums, and sums
   of squares of the publicly scaled features and the outcome). The server adds
   them, and the analyst releases N plus feature means and standard deviations,
   rounded to 4 significant digits.
2. **Round 2 (Mode B):** sites standardize their features with the released
   statistics and encrypt the 8 diagonals of their local Gram matrix plus
   `X^T y`. The server runs K = 27 encrypted gradient-descent iterations,
   refreshing the ciphertext every `L - 2` iterations, and only the 8
   coefficients are decrypted.

### Parameter sets (all 128-bit classical security, OpenFHE picks N)

| Name | N | Levels / epoch | Scale | Refreshes for K=27 | Rotation keys |
|---|---|---|---|---|---|
| P1-compact | 2^14 | 5 (3 usable) | 2^40 | 8 | shifts 1..7 |
| P2-design | 2^15 | 10 (8 usable) | 2^50 | 3 | shifts 1..7 |
| P3-deep | 2^16 | 28 (26 usable) | 2^45 | 0 | shifts 1, 2, 4 |

## Repository layout

```
src/tckks/params.py       parameter sets and public constants
src/tckks/data.py         synthetic multi-site cohort, public clipping bounds
src/tckks/baseline.py     plaintext (non-private) reference computations
src/tckks/protocol.py     the threshold-CKKS protocol (sites, server, analyst)
tests/test_correctness.py 8 correctness tests (pytest)
experiments/run_eval.py   all measurements; each configuration in its own subprocess
experiments/make_figures.py  figures + results/summary.md from results.json
report/                   evaluation report (LaTeX source + PDF)
run.sh                    one-command reproduction
Dockerfile                for macOS/Windows users
```

## Tests

`pytest -q` (about 5 s) checks that:

- the diagonal packing computes M·b correctly;
- decrypted pooled moments match the plaintext moments, and the released statistics match;
- decryption with n-1 of n shares fails;
- a collective refresh preserves the plaintext and resets the level;
- encrypted gradient descent matches plaintext gradient descent, with and without hoisted rotations, across two refreshes;
- ciphertext size does not depend on a site's record count (no size leakage);
- sites with fewer than 11 records are excluded.

## Deviations from the Part 2 design (details in the report)

1. **Two rounds with standardization.** Regressing on publicly scaled features
   (Part 2) is badly conditioned (condition number about 245) and is still
   about 97% off the exact coefficients after 27 iterations. Standardizing with
   pooled statistics from a Mode A round brings the condition number to about
   1.1. The cost is releasing feature means and standard deviations.
2. **Two reserve levels per epoch, not one.** OpenFHE's `IntMPBootAdjustScale`
   needs at least two unused levels, so P2 refreshes every 8 iterations instead
   of every 9.
3. **Noise flooding is OpenFHE's fixed default** (Gaussian, sigma = 2^20 per
   partial decryption). OpenFHE's CKKS threshold mode does not accept
   `NOISE_FLOODING_MULTIPARTY`, so flooding is not calibrated to the query
   budget as Part 2 intended.
4. **Chained key generation.** OpenFHE builds the joint public key
   sequentially (site j extends site j-1's key), not by summing independent
   shares. The resulting key has the same form, `(-a·Σ s_j + e, a)`.

## Security notes

This is a research prototype. It assumes **semi-honest** parties: the parties
run in one process and network messages are simulated, though every message is
serialized so its size is measured. There is no authentication, no malicious
security, and no protection against side channels. See the report's
limitations section.

## License

MIT for this repository's code. OpenFHE is BSD-2-Clause.
