# Evaluation conventions

Inputs are normalized [B,T,1,H,W] sequences. Predictions and observations are clipped to [0,1].

For CSI/HSS, arrays are multiplied by the evaluation scale and converted to uint16, retaining the supplied code's truncation. Input precision is retained during scaling so float32 threshold boundaries behave as before. Events use value >= threshold.

For each threshold and lead time, contingency counts are pooled across all samples and pixels. CSI is TP/(TP+FN+FP). HSS is 2*(TP*TN-FN*FP)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)). Undefined ratios become zero. Scores are averaged over lead times, then thresholds. This is not one ratio after pooling every lead time.

MSE, MAE, and RMSE use floating evaluation-scale values before integer conversion. RMSE is calculated per image before averaging. SSIM retains the original 11x11 Gaussian window (sigma 1.5), constants (0.01*scale)^2 and (0.03*scale)^2, and valid-window cropping. Image scores are averaged over samples and lead times.

The streaming evaluator stores sufficient statistics. JSON includes aggregates, per-lead-time and per-threshold values, contingency counts, sample counts, and scale metadata. Empty/nonfinite inputs are errors.

Validation computes CSI/HSS/FAR/POD; test adds SSIM/MSE/MAE/RMSE. LPIPS, PSNR, pooling scores, and the old nonstandard rank-based CRPS routine are omitted from default evaluation; these are not needed for the paper's four metrics, and no pretrained-network download is required.

Validation/test now include all real samples. Distributed sampler padding is removed with gather_for_metrics. These runner fixes may change historical aggregates if an earlier run omitted or duplicated samples.
