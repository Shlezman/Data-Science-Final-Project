# SentiSense model leaderboard (out-of-sample)

Here is the table sorted by **accuracy** in descending order:

| model [datatype] | roc_auc | f1 | accuracy |
| --- | --- | --- | --- |
| TFT [cov=none] | 0.5391 | 0.4148 | 0.5916 |
| XGBoost [embedded] | 0.5314 | 0.5129 | 0.5890 |
| XGBoost [fused] | 0.5253 | 0.4616 | 0.5759 |
| GRU [fused] | 0.5359 | 0.5238 | 0.5568 |
| PatchTST [fused] | 0.5112 | 0.3679 | 0.5553 |
| Chronos-zeroshot | 0.4266 | 0.3617 | 0.5538 |
| LSTM [embedded] | 0.5128 | 0.5317 | 0.5429 |
| XGBoost [fused] | 0.5396 | 0.5373 | 0.5417 |
| LSTM [fused] | 0.4724 | 0.4427 | 0.5402 |
| Chronos-tuned | 0.4492 | 0.4181 | 0.5381 |
| TFT [cov=scored] | 0.5524 | 0.5033 | 0.5366 |
| XGBoost [embedded] | 0.5217 | 0.5289 | 0.5347 |
| TCN [fused] | 0.5303 | 0.5280 | 0.5318 |
| TCN [scored] | 0.5669 | 0.5281 | 0.5310 |
| TFT [cov=none] | 0.5386 | 0.5212 | 0.5296 |
| GRU [scored] | 0.5755 | 0.4118 | 0.5289 |
| PatchTST [embedded] | 0.4726 | 0.3831 | 0.5283 |
| PatchTST [scored] | 0.4541 | 0.4415 | 0.5208 |
| NHiTS [cov=none] | 0.4808 | 0.4812 | 0.5157 |
| PatchTST [fused] | 0.5040 | 0.5120 | 0.5126 |
| NBEATS | 0.5227 | 0.5080 | 0.5105 |
| TCN [scored] | 0.5422 | 0.4992 | 0.5094 |
| NHiTS [cov=scored] | 0.4830 | 0.5033 | 0.5087 |
| XGBoost [scored] | 0.5338 | 0.5044 | 0.5079 |
| LSTM [scored] | 0.5204 | 0.3918 | 0.5041 |
| XGBoost [scored] | 0.5129 | 0.4997 | 0.5035 |
| PatchTST [scored] | 0.5270 | 0.4321 | 0.5035 |
| TCN [embedded] | 0.4675 | 0.4275 | 0.5022 |
| TFT [cov=scored] | 0.5119 | 0.5002 | 0.5017 |
| NBEATS | 0.5106 | 0.4980 | 0.4983 |
| LSTM [scored] | 0.5125 | 0.4938 | 0.4958 |
| GRU [embedded] | 0.5091 | 0.3414 | 0.4910 |
| NHiTS [cov=none] | 0.4837 | 0.4894 | 0.4895 |
| NHiTS [cov=scored] | 0.4835 | 0.4869 | 0.4869 |
| GRU [embedded] | 0.4642 | 0.4593 | 0.4820 |
| LSTM [fused] | 0.5115 | 0.4797 | 0.4802 |
| TCN [fused] | 0.4552 | 0.4667 | 0.4709 |
| LSTM [embedded] | 0.4715 | 0.3840 | 0.4706 |
| GRU [fused] | 0.4679 | 0.4401 | 0.4669 |
| GRU [scored] | 0.4967 | 0.4221 | 0.4644 |
| PatchTST [embedded] | 0.4552 | 0.4492 | 0.4513 |
| TCN [embedded] | 0.5327 | 0.3269 | 0.4238 |

---

**Ultimate model (best out-of-sample ROC-AUC):** `GRU [scored]` — roc_auc=0.5755, f1=0.4118

## Coverage — 23 ran, 21 cached, 2 skipped

**Ran (23):** Buy&Hold, Chronos-tuned, Chronos-zeroshot, GRU [embedded], GRU [fused], GRU [scored], LSTM [embedded], LSTM [fused], LSTM [scored], NBEATS, NHiTS [cov=none], NHiTS [cov=scored], PatchTST [embedded], PatchTST [fused], PatchTST [scored], TCN [embedded], TCN [fused], TCN [scored], TFT [cov=none], TFT [cov=scored], XGBoost [embedded], XGBoost [fused], XGBoost [scored]

**Cached (21):** Buy&Hold, GRU [embedded], GRU [fused], GRU [scored], LSTM [embedded], LSTM [fused], LSTM [scored], NBEATS, NHiTS [cov=none], NHiTS [cov=scored], PatchTST [embedded], PatchTST [fused], PatchTST [scored], TCN [embedded], TCN [fused], TCN [scored], TFT [cov=none], TFT [cov=scored], XGBoost [embedded], XGBoost [fused], XGBoost [scored]
