# 🌀 Ring Resonator Spectrum Analyzer

This Python tool provides a robust and customizable pipeline for analyzing transmission spectra of optical ring resonators. It extracts resonance wavelengths, calculates Q-factors, free spectral range (FSR), and group index by fitting dips in the transmission spectrum to Lorentzian functions.

![Example Fit](example_fit.png)

---

## 📦 Features

- ✅ Load experimental transmission spectra from `.txt` or `.wad` files
- ✅ Normalize spectra using multiple background estimation methods (`local_maxima`, `percentile`, `adaptive_window`, `moving_avg`)
- ✅ Automatically detect resonance dips
- ✅ Fit resonances using Lorentzian profiles to extract:
  - Resonance wavelength
  - FWHM linewidth
  - Q-factor
  - Fit quality (R²)
- ✅ Compute Free Spectral Range (FSR) and group index
- ✅ Tune detection parameters for narrow/broad resonances
- ✅ Visualize raw and normalized spectra with fits
- ✅ Save/export fitted results

---

## 🧪 Example Use

```python
from spectrum_analyzer import SpectrumNormalizer

   # Initialize analyzer
    ring_length = 1222
    analyzer = SpectrumNormalizer(ring_length=ring_length)
    
    # Example: Analyze a spectrum file
    file_name1 = "ex2_ringL=314.txt"
    appx_min_extinction = 0.5
    file_name2 = 'ex1_ringL=1222.txt'
    appx_min_extinction = 0.05
    skip_rows = 12 ## actual data starts here ## Skip all descrriptions and col names
    wavelength_col = 1 ## wavelegnth is in the first col (col 0 is frequency)
    transmission_col = 2 ## transmission is in the second colomn
    appx_min_extinction = 0.05 # code finds peaks below the min extinction <--- Linear scale 0.5 --> 3dB

    results = analyzer.analyze_spectrum(file_name2, convert_to_dB=True, skiprows=12, wl_col=1, trans_col=2, #if i dont want to convert to db false
                                       background_method='percentile', min_prominence=appx_min_extinction,
                                       window_size=3000,
                                       plot_debug=True)
```

---

## 🛠️ Adjustable Parameters

- `min_prominence`: Minimum depth prominence for peak detection (default = 0.01)
- `min_depth_dB`: Minimum dB depth to consider a fit valid
- `Q_bounds`: Expected Q-factor range (e.g., `(500, 200000)`)
- `gamma_bounds`: Acceptable linewidths (in nm)
- `min_r_squared`: Minimum fit quality threshold
- `offset_bounds`: Acceptable fit offset range (baseline)
- `allow_low_quality`: Whether to include imperfect fits in results

---

## 📂 Folder Structure

```
📁 spectrum-analyzer/
├── spectrum_analyzer.py      # Main analysis code
├── example_fit.png           # Example output plot
├── README.md                 # This file
└── your_data.wad             # Example input file (not tracked)
```

---

## 📊 Output

The tool returns a dictionary with:
- `resonances`: list of fitted resonance parameters
- `fsr`: estimated free spectral range (nm)
- `group_index`: calculated effective group index
- `normalized_linear`: normalized spectrum
- `background_dB`: estimated background envelope

You can extend this to export `.csv` or JSON for post-processing.

---

## 🧠 Dependencies

- `numpy`
- `matplotlib`
- `scipy`

Install them using:
```bash
pip install numpy matplotlib scipy
```

---

## 📄 License

This project is open-source

---

## 🤝 Contributions

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests to improve fitting robustness, add saving functionality, or GPU acceleration.

---

## 👨‍🔬 Author

Developed by Ramesh Kudalippalliyalil as part of research in integrated photonics and resonator characterization.  
For questions or collaboration, reach out via GitHub or email.
