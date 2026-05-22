import pandas as pd
import torch
import numpy as np
from chronos import ChronosPipeline

def main():
    print("Downloading ETTh1 dataset...")
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df = pd.read_csv(url)
    
    # Context length 512, horizon 96
    context_length = 512
    prediction_length = 96
    
    # Use the target column 'OT'
    ts = torch.tensor(df['OT'].values[:context_length], dtype=torch.float32)
    
    print("Loading amazon/chronos-t5-base model...")
    # Load Chronos pipeline. bf16 on CPU is slow/unsupported for some ops, so
    # pick the dtype from the device: bf16 on GPU, fp32 on CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map=device,
        dtype=model_dtype,
    )
    
    print("Forecasting...")
    forecast = pipeline.predict(
        inputs=ts,
        prediction_length=prediction_length,
        num_samples=20,
    )
    
    if torch.is_tensor(forecast):
        forecast = forecast.cpu()
    print(f"Forecast shape: {forecast.shape}")
    
    # Get actuals
    actuals = df['OT'].values[context_length:context_length+prediction_length]
    
    # Simple CRPS estimate
    # CRPS = E|X - y| - 0.5 * E|X - X'|
    crps_vals = []
    for i in range(prediction_length):
        samples = forecast[0, :, i].numpy()
        truth = actuals[i]
        
        mae = np.mean(np.abs(samples - truth))
        
        # pairwise absolute differences
        diffs = np.abs(samples[:, None] - samples[None, :])
        mean_diff = np.mean(diffs)
        
        crps_i = mae - 0.5 * mean_diff
        crps_vals.append(crps_i)
        
    mean_crps = np.mean(crps_vals)
    print(f"Smoke test successful! Mean CRPS on first window: {mean_crps:.4f}")

if __name__ == "__main__":
    main()
