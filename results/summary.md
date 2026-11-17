# Results summary (auto-generated)

Environment: {'python': '3.12.3', 'platform': 'Linux-6.18.44-fc-v553-x86_64-with-glibc2.39', 'cpu_count': 2, 'processor': 'x86_64', 'openfhe_python': '1.6.1.0', 'numpy': '2.5.5', 'date': '2026-11-16 19:20'}

Config: {'num_sites': 3, 'n_per_site': 5000, 'iters': 27, 'mode': 'standardized', 'trials': 3}


## Parameter sets (mean ± sd over trials)

| set | N | log Q | levels | refreshes | keygen/site s | encrypt/site s | GD total s | GD iter s | refresh total s | decrypt s | integrity s | online s | sim wall s | peak MB | Mode A rel err | beta err vs pt-GD | beta rel err vs OLS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1-compact | 16384 | 260 | 5 | 8 | 0.117 ± 0.0085 | 0.0999 ± 0.0019 | 1.37 ± 0.062 | 0.0528 ± 0.0024 | 0.093 ± 0.0085 | 0.0196 ± 0.0041 | 0.0737 ± 0.00038 | 1.66 ± 0.076 | 2.67 ± 0.073 | 363 ± 0.027 | 1.56e-07 ± 5.2e-08 | 0.000282 ± 4.6e-05 | 0.0267 ± 0.001 |
| P2-design | 32768 | 560 | 10 | 3 | 0.4 ± 0.0082 | 0.336 ± 0.017 | 4.43 ± 0.13 | 0.17 ± 0.0051 | 0.106 ± 0.0026 | 0.0572 ± 0.0075 | 0.261 ± 0.0055 | 5.19 ± 0.11 | 8.48 ± 0.067 | 1.32e+03 ± 0.055 | 2.36e-07 ± 3.5e-08 | 6.03e-07 ± 2.6e-07 | 0.0269 ± 0.0001 |
| P3-deep | 65536 | 1320 | 28 | 0 | 1.57 ± 0.058 | 1.41 ± 0.097 | 23.5 ± 1.4 | 0.906 ± 0.053 | 0 ± 0 | 0.0612 ± 0.0048 | 1.24 ± 0.095 | 26.3 ± 1.6 | 38.3 ± 2.1 | 3.86e+03 ± 0.063 | 3.46e-07 ± 1.6e-07 | 3.35e-05 ± 1.3e-05 | 0.0269 ± 0.00016 |

## Sizes (MB, first trial)

| set | public_key | ciphertext_fresh | ciphertext_out | relin_key | rot_keys_all | site_keygen_upload | modeB_site_upload | server_key_storage | refresh_share | refresh_c1_download | modeB_dec_share | modeA_dec_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1-compact | 1.57 | 1.57 | 1.05 | 6.29 | 44.05 | 58.22 | 14.17 | 50.35 | 1.18 | 0.39 | 0.53 | 0.79 |
| P2-design | 5.77 | 5.77 | 4.72 | 23.60 | 165.17 | 218.14 | 51.92 | 188.77 | 3.67 | 0.79 | 2.36 | 2.89 |
| P3-deep | 30.41 | 30.41 | 3.15 | 151.01 | 453.02 | 785.45 | 273.72 | 604.03 | 0.00 | 0.00 | 1.58 | 15.21 |

## Baseline

numpy pipeline: {'median_s': 0.0033867899999222573, 'min_s': 0.0030337610000970017, 'repeats': 20}


## Convergence

- standardized: kappa=1.1, rel err at K=27: 0.0268, K=54: 0.00072, K=120: 0.00000
- public: kappa=238.6, rel err at K=27: 0.9725, K=54: 0.96042, K=120: 0.93316
- encrypted_k54: beta rel err vs OLS 0.00073, err vs pt-GD 7.14e-07, refreshes 6, GD total 8.09s
- encrypted_public_k27: beta rel err vs OLS 0.97254, err vs pt-GD 4.81e-07, refreshes 3, GD total 4.12s

## Scaling (P2)

| n | keygen/site s | site keygen upload MB | refresh total s | decrypt s | online s | sim wall s |
|---|---|---|---|---|---|---|
| 3 | 0.378 | 218.1 | 0.096 | 0.045 | 4.60 | 7.69 |
| 5 | 0.332 | 218.1 | 0.102 | 0.048 | 4.83 | 9.30 |
| 10 | 0.331 | 218.1 | 0.112 | 0.044 | 5.17 | 13.78 |

## Ablations (P2)

- default (hoisted, all keys): GD iter 0.176s, rotation keys 165.2 MB, server key storage 188.8 MB
- no_hoist: GD iter 0.211s, rotation keys 165.2 MB, server key storage 188.8 MB
- pow2_keys: GD iter 0.184s, rotation keys 70.8 MB, server key storage 94.4 MB
