import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

class SpectrumNormalizer:

    
    def __init__(self, ring_length=5.0):
        self.L = ring_length  # μm
        self.c = 299792458  # m/s
        
    def load_spectrum(self, filename, skiprows=12, wl_col=1, trans_col=2, convert_to_dB=True):

        try:
            data = np.loadtxt(filename, skiprows=skiprows, usecols=[wl_col, trans_col])
            wavelength = data[:, 0]
            transmission = data[:, 1]
            
            if convert_to_dB:
                # Replace zero/negative values with a small positive number
                transmission_clean = np.where(transmission <= 0, 1e-10, transmission)
                transmission_dB = 10 * np.log10(transmission_clean)
            else:
                transmission_dB = transmission
                
            print(f"Loaded spectrum: {len(wavelength)} points")
            print(f"Wavelength range: {wavelength.min():.2f} - {wavelength.max():.2f} nm")
            print(f"Transmission range: {transmission_dB.min():.2f} - {transmission_dB.max():.2f} dB")
            
            return wavelength, transmission_dB
            
        except Exception as e:
            print(f"Error loading spectrum: {e}")
            return None, None
    
    def calculate_background_envelope(self, wavelength, transmission_dB, method='local_maxima', window_size=50):

        if method == 'local_maxima':
            # Find local maxima as background points, then interpolate
            peaks, _ = find_peaks(transmission_dB, distance=window_size//2, prominence=1.0)
            
            if len(peaks) < 2:
                print("Warning: Not enough local maxima found, falling back to percentile method")
                return self.calculate_background_envelope(wavelength, transmission_dB, 
                                                        method='percentile', window_size=window_size)
            
            # Interpolate between maxima to get smooth background
            peak_wavelengths = wavelength[peaks]
            peak_values = transmission_dB[peaks]
            
            # Add endpoints to ensure full coverage
            if peaks[0] > 10:
                peak_wavelengths = np.concatenate([[wavelength[0]], peak_wavelengths])
                peak_values = np.concatenate([[transmission_dB[0]], peak_values])
            if peaks[-1] < len(wavelength) - 10:
                peak_wavelengths = np.concatenate([peak_wavelengths, [wavelength[-1]]])
                peak_values = np.concatenate([peak_values, [transmission_dB[-1]]])
            
            interp_func = interp1d(peak_wavelengths, peak_values, kind='linear', 
                                 fill_value='extrapolate')
            background_dB = interp_func(wavelength)
            
        elif method == 'percentile':
            # Use rolling percentile as background
            percentile = 90  # Use 90th percentile as background
            background_dB = np.zeros_like(transmission_dB)
            
            half_window = window_size // 2
            for i in range(len(transmission_dB)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(transmission_dB), i + half_window + 1)
                window_data = transmission_dB[start_idx:end_idx]
                background_dB[i] = np.percentile(window_data, percentile)
            
            # Smooth the percentile background
            background_dB = uniform_filter1d(background_dB, size=20, mode='reflect')
            
        elif method == 'adaptive_window':
            # Use adaptive window that adjusts based on local signal variation
            min_window = 20
            max_window = window_size
            background_dB = np.zeros_like(transmission_dB)
            
            for i in range(len(transmission_dB)):
                # Adaptive window based on local variation
                local_std = np.std(transmission_dB[max(0, i-50):min(len(transmission_dB), i+50)])
                adaptive_window = int(min_window + (max_window - min_window) * min(local_std/5.0, 1.0))
                
                half_window = adaptive_window // 2
                start_idx = max(0, i - half_window)
                end_idx = min(len(transmission_dB), i + half_window + 1)
                window_data = transmission_dB[start_idx:end_idx]
                background_dB[i] = np.percentile(window_data, 85)  # 85th percentile
            
            # Smooth the result
            background_dB = uniform_filter1d(background_dB, size=10, mode='reflect')
            
        else:  # method == 'moving_avg'
            background_dB = uniform_filter1d(transmission_dB, size=window_size, mode='reflect')
        
        # Ensure background is always >= original signal
        offset_dB = 0.5  # Small offset to ensure background is above signal
        min_background = np.maximum(transmission_dB + offset_dB, background_dB)
        background_dB = uniform_filter1d(min_background, size=10, mode='reflect')
        
        print(f"Background calculation method: {method}")
        print(f"Window size: {window_size} points")
        print(f"Background range: {np.min(background_dB):.1f} to {np.max(background_dB):.1f} dB")
        
        return background_dB
    
    def normalize_spectrum(self, wavelength, transmission_dB, background_method='local_maxima', 
                          window_size=50, plot_debug=False):

        # Calculate background envelope
        background_dB = self.calculate_background_envelope(wavelength, transmission_dB, 
                                                          method=background_method, 
                                                          window_size=window_size)
        
        # Subtract background (in dB domain)
        normalized_dB = transmission_dB - background_dB

        print(np.min(normalized_dB))
        
        # Convert to linear scale (normalized_linear <= 1.0, resonances as dips)
        normalized_linear = 10**(normalized_dB/10)
        
        print(f"Normalized dB range: {np.min(normalized_dB):.3f} to {np.max(normalized_dB):.3f} dB")
        print(f"Normalized linear range: {np.min(normalized_linear):.6f} to {np.max(normalized_linear):.6f}")
        
        if plot_debug:
            self.plot_normalization_debug(wavelength, transmission_dB, background_dB, 
                                        normalized_dB, normalized_linear, background_method)
        
        return wavelength, normalized_linear, background_dB, normalized_dB
    
    def plot_normalization_debug(self, wavelength, transmission_dB, background_dB, 
                               normalized_dB, normalized_linear, method_name):
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Original and background
        ax1.plot(wavelength, transmission_dB, 'b-', alpha=0.7, label='Original')
        ax1.plot(wavelength, background_dB, 'r-', alpha=0.8, label=f'Background ({method_name})')
        ax1.set_ylabel('Transmission (dB)')
        ax1.set_title('Background Envelope Extraction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Normalized (dB)
        ax2.plot(wavelength, normalized_dB, 'g-', alpha=0.7)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5, label='Baseline (0 dB)')
        ax2.set_ylabel('Normalized Transmission (dB)')
        ax2.set_title('After Background Subtraction (dB scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Normalized (linear)
        ax3.plot(wavelength, normalized_linear, 'm-', alpha=0.7)
        ax3.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Baseline (1.0)')
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Normalized Transmission (linear)')
        ax3.set_title('Normalized Spectrum (Linear scale) - Resonances as dips')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def lorentzian_dip(self, x, x0, gamma, A, offset):
        ####Lorentzian for fitting resonance dips in normalized linear transmission
        return offset - A * gamma**2 / ((x - x0)**2 + gamma**2)
        
    def find_resonances(self, wavelength, normalized_linear,
                        min_prominence=0.8, min_depth_dB=1.0, plot_debug=False,
                        Q_bounds=(500, 200000), gamma_bounds=(0.005, 1.0),
                        min_r_squared=0.6, offset_bounds=(0.5, 1.5),
                        allow_low_quality=False):
        
        # Find dips by inverting signal (dips become peaks)
        inverted_signal = 1.0 - normalized_linear
        
        # Dynamic thresholds
        signal_std = np.std(inverted_signal)
        signal_mean = np.mean(inverted_signal)
        adaptive_prominence = max(min_prominence, 2 * signal_std)
        adaptive_height = signal_mean + signal_std
        
        # Minimum distance between peaks (estimate FSR)
        wavelength_step = np.mean(np.diff(wavelength))
        min_distance = max(10, int(0.1 / wavelength_step))  # At least 0.1 nm separation
        
        peaks, properties = find_peaks(inverted_signal, 
                                     prominence=adaptive_prominence,
                                     distance=min_distance,
                                     height=adaptive_height,
                                     width=3)
        
        print(f"Found {len(peaks)} potential resonances")
        print(f"Peak detection - prominence: {adaptive_prominence:.4f}, height: {adaptive_height:.4f}")
        
        if plot_debug and len(peaks) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(wavelength, inverted_signal, 'b-', alpha=0.7, label='Inverted signal')
            plt.plot(wavelength[peaks], inverted_signal[peaks], 'ro', markersize=8, label='Detected peaks')
            plt.axhline(adaptive_height, color='r', linestyle='--', alpha=0.5, label='Height threshold')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('1 - Normalized Transmission')
            plt.title('Peak Detection in Inverted Signal')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        resonances = []
        
        for i, peak_idx in enumerate(peaks):
            try:
                # Adaptive fitting window
                if 'widths' in properties:
                    peak_width = properties['widths'][i]
                    window = max(30, int(3 * peak_width))
                else:
                    window = 40
                
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(wavelength), peak_idx + window)
                
                x_fit = wavelength[start_idx:end_idx]
                y_fit = normalized_linear[start_idx:end_idx]
                
                # Initial parameter estimates
                x0_guess = wavelength[peak_idx]
                peak_min_val = y_fit[peak_idx - start_idx]
                baseline_est = np.percentile(y_fit, 90)
                depth_est = baseline_est - peak_min_val
                
                # Estimate gamma from FWHM
                half_max = peak_min_val + depth_est/2
                half_max_indices = np.where(y_fit <= half_max)[0]
                if len(half_max_indices) >= 2:
                    fwhm_est = x_fit[half_max_indices[-1]] - x_fit[half_max_indices[0]]
                    gamma_guess = fwhm_est / 2
                else:
                    gamma_guess = 0.05
                
                gamma_guess = np.clip(gamma_guess, 0.01, 0.5)
                A_guess = depth_est
                offset_guess = baseline_est
                
                # Fit with bounds
                bounds = ([x0_guess - 0.5, 0.005, 0.001, 0.5],
                         [x0_guess + 0.5, 0.5, 1.0, 1.5])
                
                popt, pcov = curve_fit(self.lorentzian_dip, x_fit, y_fit,
                                     p0=[x0_guess, gamma_guess, A_guess, offset_guess],
                                     bounds=bounds, maxfev=5000)
                
                x0, gamma, A, offset = popt
                Q = x0 / (2 * gamma)  # Q = λ/FWHM
                
                # Calculate errors
                param_errors = np.sqrt(np.diag(pcov))
                Q_error = Q * np.sqrt((param_errors[0]/x0)**2 + (param_errors[1]/gamma)**2)
                
                # Fit quality
                y_fitted = self.lorentzian_dip(x_fit, *popt)
                residuals = y_fit - y_fitted
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Calculate depth in dB
                min_normalized = offset - A
                if min_normalized > 0:
                    depth_dB = -10 * np.log10(min_normalized)
                else:
                    depth_dB = 0
                
                # Quality checks
                quality_checks = {
                        'Q_reasonable': Q_bounds[0] < Q < Q_bounds[1],
                        'depth_sufficient': depth_dB >= min_depth_dB,
                        'fit_quality': r_squared > min_r_squared,
                        'offset_reasonable': offset_bounds[0] < offset < offset_bounds[1],
                        'amplitude_reasonable': A > min_prominence,
                        'gamma_reasonable': gamma_bounds[0] < gamma < gamma_bounds[1]
                    }

                
                if all(quality_checks.values()) or allow_low_quality:
                    resonances.append({
                        'wavelength': x0,
                        'wavelength_error': param_errors[0],
                        'linewidth_fwhm': 2 * gamma,
                        'linewidth_error': 2 * param_errors[1],
                        'Q': Q,
                        'Q_error': Q_error,
                        'amplitude': A,
                        'offset': offset,
                        'depth_dB': depth_dB,
                        'fit_params': popt,
                        'fit_errors': param_errors,
                        'fit_quality': r_squared,
                        'quality_checks': quality_checks
                    })
                    print(f"✓ Resonance {len(resonances)}: λ={x0:.3f}±{param_errors[0]:.3f} nm, "
                          f"Q={Q:.0f}±{Q_error:.0f}, depth={depth_dB:.1f} dB, R²={r_squared:.3f}")
                else:
                    failed_checks = [k for k, v in quality_checks.items() if not v]
                    print(f"✗ Rejected resonance at {x0:.3f} nm: failed {failed_checks}")
                
            except Exception as e:
                print(f"✗ Failed to fit resonance {i+1}: {e}")
                continue
        
        print(f"Successfully fitted {len(resonances)} high-quality resonances")
        return resonances
    
    def calculate_fsr_and_ng(self, resonances, target_wavelength=1550):
        
        if len(resonances) < 2:
            print("Need at least 2 resonances to calculate FSR")
            return None, None
            
        wavelengths = [r['wavelength'] for r in resonances]
        wavelengths.sort()
        
        # Calculate FSR (spacing between adjacent resonances)
        fsrs = np.diff(wavelengths)
        avg_fsr = np.mean(fsrs)
        fsr_std = np.std(fsrs) if len(fsrs) > 1 else 0
        
        # Calculate group index: ng = λ²/(2πR × FSR)
        wavelength_m = target_wavelength * 1e-9
        fsr_m = avg_fsr * 1e-9
        L_m = self.L * 1e-6
        
        ng = wavelength_m**2 / (L_m * fsr_m)
        
        print(f"FSR: {avg_fsr:.3f} ± {fsr_std:.3f} nm")
        print(f"Group index (ng): {ng:.3f}")
        
        return avg_fsr, ng
    
    def analyze_spectrum(self, filename,convert_to_dB=True,skiprows=12, wl_col=1, trans_col=2, background_method='local_maxima', window_size=3000, 
                        min_prominence=0.01, plot_debug=True):
        
        # Load spectrum
        wavelength, transmission_dB = self.load_spectrum(filename,skiprows, wl_col, trans_col,convert_to_dB)
        if wavelength is None:
            return None
        
        # Normalize spectrum
        print("\nNormalizing spectrum...")
        wl, norm_linear, background_dB, norm_dB = self.normalize_spectrum(
            wavelength, transmission_dB, background_method=background_method, 
            window_size=window_size, plot_debug=plot_debug)
        
        # Find resonances
        print("\nFinding resonances...")
        resonances = self.find_resonances(
                    wl, norm_linear, min_prominence=min_prominence,
                    min_depth_dB=1.0, plot_debug=plot_debug,
                    Q_bounds=(500, 200000), gamma_bounds=(0.005, 1.0),
                    min_r_squared=0.6, offset_bounds=(0.5, 1.5),
                    allow_low_quality=True  # Let you inspect all fits
                )

        
        # Calculate FSR and group index
        print("\nCalculating FSR and group index...")
        fsr, ng = self.calculate_fsr_and_ng(resonances)
        
        # Plot results
        if plot_debug:
            self.plot_results(wavelength, transmission_dB, norm_linear, resonances)
        
        results = {
            'wavelength': wavelength,
            'transmission_dB': transmission_dB,
            'normalized_linear': norm_linear,
            'background_dB': background_dB,
            'resonances': resonances,
            'fsr': fsr,
            'group_index': ng,
            'n_resonances': len(resonances),
            'analysis_params': {
                'background_method': background_method,
                'window_size': window_size,
                'min_prominence': min_prominence
            }
        }
        
        # Print summary
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"File: {filename}")
        print(f"Ring radius: {self.L} μm")
        print(f"Background method: {background_method}")
        print(f"Number of resonances found: {len(resonances)}")
        if fsr:
            print(f"FSR: {fsr:.3f} nm")
            print(f"Group index: {ng:.3f}")
        
        if resonances:
            print(f"\nResonance details:")
            for i, res in enumerate(resonances):
                print(f"  Resonance {i+1}: λ={res['wavelength']:.3f}±{res['wavelength_error']:.3f} nm, "
                      f"Q={res['Q']:.0f}±{res['Q_error']:.0f}, depth={res['depth_dB']:.1f} dB")
        
        return results
    
    def plot_results(self, wavelength, transmission_dB, normalized_linear, resonances):
        """Plot analysis results"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original spectrum
        ax1.plot(wavelength, transmission_dB, 'b-', alpha=0.7, label='Original spectrum')
        for res in resonances:
            ax1.axvline(res['wavelength'], color='red', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Transmission (dB)')
        ax1.set_title('Original Spectrum with Detected Resonances')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Normalized spectrum with fits
        ax2.plot(wavelength, normalized_linear, 'b-', alpha=0.7, label='Normalized spectrum')
        
        # Plot fitted resonances
        for i, res in enumerate(resonances):
            wl_res = res['wavelength']
            params = res['fit_params']
            
            # Create fit curve around resonance
            wl_range = np.linspace(wl_res - 0.5, wl_res + 0.5, 100)
            fit_curve = self.lorentzian_dip(wl_range, *params)
            
            ax2.plot(wl_range, fit_curve, 'r-', alpha=0.8, linewidth=2)
            ax2.plot(wl_res, res['offset'] - res['amplitude'], 'ro', markersize=6)
            
        ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Baseline')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Normalized Transmission')
        ax2.set_title('Normalized Spectrum with Lorentzian Fits')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def test_background_methods(self, filename):
        
        wavelength, transmission_dB = self.load_spectrum(filename)
        if wavelength is None:
            return
        
        methods = ['local_maxima', 'percentile', 'adaptive_window', 'moving_avg']
        window_sizes = [3000, 3000, 5000]
        
        fig, axes = plt.subplots(len(methods), len(window_sizes), figsize=(15, 12))
        
        for i, method in enumerate(methods):
            for j, window_size in enumerate(window_sizes):
                ax = axes[i, j] if len(methods) > 1 else axes[j]
                
                try:
                    background_dB = self.calculate_background_envelope(
                        wavelength, transmission_dB, method=method, window_size=window_size)
                    
                    ax.plot(wavelength, transmission_dB, 'b-', alpha=0.7, label='Original')
                    ax.plot(wavelength, background_dB, 'r-', alpha=0.8, label='Background')
                    
                    is_valid = np.all(background_dB >= transmission_dB - 0.1)
                    title_color = 'green' if is_valid else 'red'
                    
                    ax.set_title(f'{method}\nwindow={window_size}\nValid: {is_valid}', 
                               color=title_color, fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    if i == 0 and j == 0:
                        ax.legend()
                        
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{method}\nwindow={window_size}\nERROR', color='red')
        
        plt.tight_layout()
        plt.show()
        
        print("Background method test complete!")
        print("Green titles = valid background, Red titles = invalid/error")

# Example usage
if __name__ == "__main__":
    import os
    script_path = os.path.abspath(__file__)


    script_dir = os.path.dirname(script_path)

    os.chdir(script_dir)
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

     
    # Example: Test different background methods
    # analyzer.test_background_methods('Ring2_circ_5um.wad')
    
    print("Spectrum analyzer ready :-)!")
    print("Usage:")
    print("1. results = analyzer.analyze_spectrum('filename.txt')")
    print("2. analyzer.test_background_methods('filename.txt')")
    print("3. Adjust background_method and window_size based on your data")