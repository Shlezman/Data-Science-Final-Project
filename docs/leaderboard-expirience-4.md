# SentiSense model leaderboard (out-of-sample)

| model [datatype/regime] | roc_auc | f1 | mcc | accuracy | cum_return | sharpe | max_drawdown | n |
|---|---|---|---|---|---|---|---|---|
| XGBoost [embedded/CUT] | 0.5217 | 0.5289 | 0.0748 | 0.5347 | 0.0279 | 0.2515 | -0.1 | 288 |
| LSTM [embedded/CUT] | 0.4715 | 0.384 | -0.0658 | 0.4706 | -0.1391 | -0.8154 | -0.1943 | 272 |
| GRU [embedded/CUT] | 0.5091 | 0.3414 | 0.0322 | 0.491 | -0.1023 | -0.5265 | -0.1986 | 277 |
| TCN [embedded/CUT] | 0.4675 | 0.4275 | -0.0154 | 0.5022 | -0.0352 | -0.1755 | -0.1481 | 227 |
| PatchTST [embedded/CUT] | 0.4552 | 0.4492 | -0.0948 | 0.4513 | -0.1938 | -1.6143 | -0.2129 | 277 |
| XGBoost [fused/CUT] | 0.5396 | 0.5373 | 0.0882 | 0.5417 | 0.0792 | 0.5817 | -0.0832 | 288 |
| LSTM [fused/CUT] | 0.5115 | 0.4797 | -0.0379 | 0.4802 | 0.027 | 0.3442 | -0.0584 | 227 |
| GRU [fused/CUT] | 0.4679 | 0.4401 | -0.0617 | 0.4669 | -0.0383 | -0.5209 | -0.0935 | 242 |
| TCN [fused/CUT] | 0.5303 | 0.528 | 0.0615 | 0.5318 | -0.0132 | -0.0426 | -0.0956 | 267 |
| PatchTST [fused/CUT] | 0.504 | 0.512 | 0.0241 | 0.5126 | -0.0889 | -0.7168 | -0.1861 | 277 |
| XGBoost [scored/CUT] | 0.5129 | 0.4997 | 0.0097 | 0.5035 | 0.0503 | 0.4208 | -0.1273 | 288 |
| LSTM [scored/CUT] | 0.5204 | 0.3918 | -0.0226 | 0.5041 | -0.0344 | -0.1617 | -0.1481 | 242 |
| GRU [scored/CUT] | 0.5755 | 0.4118 | 0.0731 | 0.5289 | 0.0125 | 0.161 | -0.1481 | 242 |
| TCN [scored/CUT] | 0.5422 | 0.4992 | 0.0143 | 0.5094 | 0.0386 | 0.3838 | -0.0915 | 267 |
| PatchTST [scored/CUT] | 0.527 | 0.4321 | 0.0251 | 0.5035 | -0.0463 | -0.2005 | -0.1493 | 282 |
| Buy&Hold [CUT] | 0.5 | 0.3302 | 0 | 0.4931 | -0.0632 | -0.2723 | -0.1986 | 288 |
| Chronos-zeroshot | 0.4266 | 0.3617 | -0.123 | 0.5538 | 0.8604 | 2.5183 | -0.1299 | 381 |
| Chronos-tuned | 0.4492 | 0.4181 | -0.0666 | 0.5381 | 0.8413 | 2.7427 | -0.0984 | 381 |
| TFT [cov=scored/CUT] | 0.5119 | 0.5002 | 0.0024 | 0.5017 | -0.0511 | -0.3705 | -0.144 | 287 |
| TFT [cov=none/CUT] | 0.5386 | 0.5212 | 0.0646 | 0.5296 | -0.0098 | -0.0071 | -0.1471 | 287 |
| NHiTS [cov=scored/CUT] | 0.483 | 0.5033 | 0.0157 | 0.5087 | 0.0046 | 0.0901 | -0.1153 | 287 |
| NHiTS [cov=none/CUT] | 0.4808 | 0.4812 | 0.0302 | 0.5157 | 0.0334 | 0.4244 | -0.0659 | 287 |
| NBEATS [CUT] | 0.5106 | 0.498 | -0.0029 | 0.4983 | 0.0446 | 0.3816 | -0.1136 | 287 |
| XGBoost [scored/FULL] | 0.5338 | 0.5044 | 0.0142 | 0.5079 | 0.4686 | 2.052 | -0.0793 | 382 |
| LSTM [scored/FULL] | 0.5125 | 0.4938 | 0.0336 | 0.4958 | 0.3008 | 1.5809 | -0.0896 | 361 |
| GRU [scored/FULL] | 0.4967 | 0.4221 | 0.0397 | 0.4644 | 0.077 | 0.6309 | -0.0718 | 351 |
| TCN [scored/FULL] | 0.5669 | 0.5281 | 0.1138 | 0.531 | 0.5118 | 2.8001 | -0.0472 | 371 |
| PatchTST [scored/FULL] | 0.4541 | 0.4415 | -0.0604 | 0.5208 | 0.6903 | 2.3295 | -0.0913 | 361 |
| XGBoost [embedded/FULL] | 0.5314 | 0.5129 | 0.1027 | 0.589 | 0.9734 | 2.9418 | -0.09 | 382 |
| LSTM [embedded/FULL] | 0.5128 | 0.5317 | 0.0634 | 0.5429 | 0.6421 | 2.9397 | -0.071 | 361 |
| GRU [embedded/FULL] | 0.4642 | 0.4593 | -0.0782 | 0.482 | 0.3951 | 1.6386 | -0.1043 | 361 |
| TCN [embedded/FULL] | 0.5327 | 0.3269 | -0.0462 | 0.4238 | -0.0087 | -0.1117 | -0.0544 | 361 |
| PatchTST [embedded/FULL] | 0.4726 | 0.3831 | -0.1159 | 0.5283 | 0.8414 | 2.5736 | -0.09 | 371 |
| XGBoost [fused/FULL] | 0.5253 | 0.4616 | 0.0499 | 0.5759 | 0.9529 | 2.8086 | -0.09 | 382 |
| LSTM [fused/FULL] | 0.4724 | 0.4427 | -0.0292 | 0.5402 | 0.6969 | 2.3443 | -0.1136 | 361 |
| GRU [fused/FULL] | 0.5359 | 0.5238 | 0.0632 | 0.5568 | 0.8494 | 3.0174 | -0.0814 | 361 |
| TCN [fused/FULL] | 0.4552 | 0.4667 | -0.0113 | 0.4709 | 0.3925 | 2.0778 | -0.082 | 361 |
| PatchTST [fused/FULL] | 0.5112 | 0.3679 | -0.0863 | 0.5553 | 0.938 | 2.7266 | -0.09 | 371 |
| Buy&Hold [FULL] | 0.5 | 0.3665 | 0 | 0.5785 | 1.0349 | 2.8214 | -0.09 | 382 |
| TFT [cov=scored/FULL] | 0.5524 | 0.5033 | 0.018 | 0.5366 | 0.7467 | 2.4856 | -0.09 | 382 |
| TFT [cov=none/FULL] | 0.5391 | 0.4148 | 0.1066 | 0.5916 | 1.1226 | 3.031 | -0.09 | 382 |
| NHiTS [cov=scored/FULL] | 0.4835 | 0.4869 | 0.0001 | 0.4869 | 0.5173 | 2.7658 | -0.0444 | 382 |
| NHiTS [cov=none/FULL] | 0.4837 | 0.4894 | -0.0004 | 0.4895 | 0.5391 | 2.7805 | -0.0516 | 382 |
| NBEATS [FULL] | 0.5227 | 0.508 | 0.0237 | 0.5105 | 0.5348 | 2.4245 | -0.0675 | 382 |

**Ultimate model (best out-of-sample ROC-AUC):** `GRU [scored/CUT]` — roc_auc=0.5755, f1=0.4118, mcc=0.0731, sharpe=0.1610, cum_return=0.0125

## Coverage — 23 ran, 21 cached, 2 skipped

**Ran (23):** `Buy&Hold [FULL]`, `Chronos-tuned`, `Chronos-zeroshot`, `GRU [embedded/FULL]`, `GRU [fused/FULL]`, `GRU [scored/FULL]`, `LSTM [embedded/FULL]`, `LSTM [fused/FULL]`, `LSTM [scored/FULL]`, `NBEATS [FULL]`, `NHiTS [cov=none/FULL]`, `NHiTS [cov=scored/FULL]`, `PatchTST [embedded/FULL]`, `PatchTST [fused/FULL]`, `PatchTST [scored/FULL]`, `TCN [embedded/FULL]`, `TCN [fused/FULL]`, `TCN [scored/FULL]`, `TFT [cov=none/FULL]`, `TFT [cov=scored/FULL]`, `XGBoost [embedded/FULL]`, `XGBoost [fused/FULL]`, `XGBoost [scored/FULL]`

**Cached (21):** `Buy&Hold [CUT]`, `GRU [embedded/CUT]`, `GRU [fused/CUT]`, `GRU [scored/CUT]`, `LSTM [embedded/CUT]`, `LSTM [fused/CUT]`, `LSTM [scored/CUT]`, `NBEATS [CUT]`, `NHiTS [cov=none/CUT]`, `NHiTS [cov=scored/CUT]`, `PatchTST [embedded/CUT]`, `PatchTST [fused/CUT]`, `PatchTST [scored/CUT]`, `TCN [embedded/CUT]`, `TCN [fused/CUT]`, `TCN [scored/CUT]`, `TFT [cov=none/CUT]`, `TFT [cov=scored/CUT]`, `XGBoost [embedded/CUT]`, `XGBoost [fused/CUT]`, `XGBoost [scored/CUT]`

**Skipped (why):**
- `TimesFM [CUT]` — The 'timesfm' extra is not installed. On the server:
    uv sync --extra timesfm   (or: pip install 'timesfm[torch]')
    # if PyP
- `TimesFM [FULL]` — The 'timesfm' extra is not installed. On the server:
    uv sync --extra timesfm   (or: pip install 'timesfm[torch]')
    # if PyP
