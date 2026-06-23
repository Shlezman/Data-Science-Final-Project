SentiSense model leaderboard (out-of-sample)

model [datatype]roc_aucf1accuracyXGBoost [embedded]0.52170.52890.5347LSTM [embedded]0.47150.3840.4706GRU [embedded]0.50910.34140.491TCN [embedded]0.46750.42750.5022PatchTST [embedded]0.45520.44920.4513XGBoost [fused]0.53960.53730.5417LSTM [fused]0.51150.47970.4802GRU [fused]0.46790.44010.4669TCN [fused]0.53030.5280.5318PatchTST [fused]0.5040.5120.5126XGBoost [scored]0.51290.49970.5035LSTM [scored]0.52040.39180.5041GRU [scored]0.57550.41180.5289TCN [scored]0.54220.49920.5094PatchTST [scored]0.5270.43210.5035Buy&Hold0.50.33020.4931Chronos-zeroshot0.42660.36170.5538Chronos-tuned0.44920.41810.5381TFT [cov=scored]0.51190.50020.5017TFT [cov=none]0.53860.52120.5296NHiTS [cov=scored]0.4830.50330.5087NHiTS [cov=none]0.48080.48120.5157NBEATS0.51060.4980.4983XGBoost [scored]0.53380.50440.5079LSTM [scored]0.51250.49380.4958GRU [scored]0.49670.42210.4644TCN [scored]0.56690.52810.531PatchTST [scored]0.45410.44150.5208XGBoost [embedded]0.53140.51290.589LSTM [embedded]0.51280.53170.5429GRU [embedded]0.46420.45930.482TCN [embedded]0.53270.32690.4238PatchTST [embedded]0.47260.38310.5283XGBoost [fused]0.52530.46160.5759LSTM [fused]0.47240.44270.5402GRU [fused]0.53590.52380.5568TCN [fused]0.45520.46670.4709PatchTST [fused]0.51120.36790.5553Buy&Hold0.50.36650.5785TFT [cov=scored]0.55240.50330.5366TFT [cov=none]0.53910.41480.5916NHiTS [cov=scored]0.48350.48690.4869NHiTS [cov=none]0.48370.48940.4895NBEATS0.52270.5080.5105

Ultimate model (best out-of-sample ROC-AUC): GRU [scored] — roc_auc=0.5755, f1=0.4118

Coverage — 23 ran, 21 cached, 2 skipped

Ran (23): Buy&Hold, Chronos-tuned, Chronos-zeroshot, GRU [embedded], GRU [fused], GRU [scored], LSTM [embedded], LSTM [fused], LSTM [scored], NBEATS, NHiTS [cov=none], NHiTS [cov=scored], PatchTST [embedded], PatchTST [fused], PatchTST [scored], TCN [embedded], TCN [fused], TCN [scored], TFT [cov=none], TFT [cov=scored], XGBoost [embedded], XGBoost [fused], XGBoost [scored]

Cached (21): Buy&Hold, GRU [embedded], GRU [fused], GRU [scored], LSTM [embedded], LSTM [fused], LSTM [scored], NBEATS, NHiTS [cov=none], NHiTS [cov=scored], PatchTST [embedded], PatchTST [fused], PatchTST [scored], TCN [embedded], TCN [fused], TCN [scored], TFT [cov=none], TFT [cov=scored], XGBoost [embedded], XGBoost [fused], XGBoost [scored]

