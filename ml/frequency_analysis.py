import cv2
import numpy as np

def analyze_frequency_domain(face_image_bgr):
    """
    FFT-based frequency domain analysis using OpenCV and NumPy.
    Extracts magnitude spectrum and calculates spectral energy anomaly score.
    
    Returns:
    - anomaly_score: float (0.0 to 1.0)
    - frequency_evidence: str description
    - spectrum_visual_bgr: uint8 BGR image of log magnitude spectrum
    """
    if face_image_bgr is None or face_image_bgr.size == 0:
        return 0.0, "Invalid image for frequency analysis", None
        
    # Convert image to grayscale for frequency analysis
    gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 2D Fast Fourier Transform
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    
    # Calculate Magnitude Spectrum
    magnitude_spectrum = np.abs(dft_shift)
    log_spectrum = np.log(1 + magnitude_spectrum)
    
    # Normalize spectrum for visualization (0 to 255)
    spectrum_norm = cv2.normalize(log_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    spectrum_visual_bgr = cv2.applyColorMap(spectrum_norm, cv2.COLORMAP_VIRIDIS)
    
    # Compute Radial Power Distribution (Azimuthal Average)
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_radius = int(min(center_y, center_x))
    
    if max_radius <= 0:
        return 0.0, "Face region too small for frequency decomposition", spectrum_visual_bgr
        
    # Radial bins: Low frequency (r < 0.25*R), High frequency (r > 0.65*R)
    low_freq_mask = r < (0.25 * max_radius)
    high_freq_mask = (r > (0.65 * max_radius)) & (r <= max_radius)
    
    low_energy = np.sum(magnitude_spectrum[low_freq_mask])
    high_energy = np.sum(magnitude_spectrum[high_freq_mask])
    
    total_energy = np.sum(magnitude_spectrum) + 1e-8
    high_energy_ratio = high_energy / total_energy
    
    # Standard natural face images have smooth 1/f spectral decay.
    # Deepfakes often display anomalous high-frequency grid noise or unusual high energy ratios.
    # Typical baseline for natural faces: ratio ~ 0.05 to 0.15.
    baseline_high_freq = 0.12
    anomaly_factor = max(0.0, high_energy_ratio - baseline_high_freq) / (0.25 - baseline_high_freq)
    anomaly_score = float(np.clip(anomaly_factor, 0.0, 1.0))
    
    if anomaly_score > 0.4:
        frequency_evidence = f"Frequency-domain characteristics differ from expected natural-image patterns (High-Frequency Spectral Anomaly: {int(anomaly_score * 100)}%)"
    else:
        frequency_evidence = f"Frequency spectral distribution aligns with natural facial photography standards (Anomaly Score: {int(anomaly_score * 100)}%)"
        
    return anomaly_score, frequency_evidence, spectrum_visual_bgr
