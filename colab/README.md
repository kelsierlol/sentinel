# Sentinel Phase 2: Google Colab Benchmark

This folder contains everything you need to run Phase 2 benchmarks on Google Colab.

## Quick Start

1. **Open the notebook in Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `Phase2_ImageNetC_Benchmark.ipynb`
   - OR use this direct link: `https://colab.research.google.com/github/kelsierlol/sentinel/blob/main/colab/Phase2_ImageNetC_Benchmark.ipynb`

2. **Set runtime to GPU:**
   - Runtime → Change runtime type → T4 GPU

3. **Run all cells:**
   - Runtime → Run all

## What It Does

- ✅ Installs Sentinel from GitHub
- ✅ Trains on CIFAR-10 (10-15 mins)
- ✅ Downloads ImageNet-C subset
- ✅ Benchmarks on 3 corruption types (~15 mins)
- ✅ Compares against OpenCV baselines
- ✅ Generates comparison plots
- ✅ Downloads results JSON

## Files

- `Phase2_ImageNetC_Benchmark.ipynb` - Main notebook
- `phase2_benchmark.py` - Benchmark implementation
- `README.md` - This file

## Requirements

- Google Colab (free tier works!)
- GPU runtime (T4 recommended)
- ~30-45 minutes runtime

## Quick Test (3 corruptions)

Default configuration tests:
- `gaussian_noise`
- `defocus_blur`
- `motion_blur`

This gives you a quick validation in ~30 minutes.

## Full Benchmark (15 corruptions)

Uncomment the full benchmark cell to test all 15 ImageNet-C corruption types.

**Warning:** Takes ~2 hours, uses more GPU quota.

## ImageNet-C Dataset

### Option 1: Download from Kaggle (Recommended)

The notebook includes instructions to download ImageNet-C from Kaggle.

You'll need:
1. Kaggle account
2. `kaggle.json` API token

### Option 2: Use Google Drive

If you already have ImageNet-C:
1. Upload to Google Drive
2. Mount Drive in Colab
3. Update paths in notebook

### Option 3: Manual Download

Download from: https://zenodo.org/record/2235448

## Expected Results

**Target:** Beat OpenCV baselines by 10-15%

**Example output:**
```
Sentinel mean AUROC:     0.8500
OpenCV Blur mean AUROC:  0.7200
Improvement:             +18.1%

✅ SUCCESS: Beat OpenCV by 18.1%
```

## Troubleshooting

### "Out of GPU memory"
- Reduce `max_samples_per_corruption` to 250
- Use smaller batch size

### "ImageNet-C not found"
- Check dataset paths
- Make sure you've downloaded ImageNet-C
- Try the Kaggle download method

### "Training takes too long"
- Reduce epochs: `epochs_unet=5, epochs_alpha=5`
- Use fewer samples for training

## Output Files

After running:
- `benchmark_results.json` - Raw results
- `benchmark_comparison.png` - Visualization
- Both auto-download at the end

## Next Steps

After Phase 2 completes:
1. ✅ Verify improvement > 10%
2. ✅ Download results
3. ✅ Add to Sentinel README
4. Move to Phase 3: Full CLI implementation

---

**Need help?** Open an issue on GitHub: https://github.com/kelsierlol/sentinel/issues
