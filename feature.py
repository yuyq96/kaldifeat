import math

import numpy as np
from scipy.fftpack import dct

from .window import sliding_window, extract_window


def compute_fbank_feats(
        waveform,
        blackman_coeff=0.42,
        dither=1.0,
        energy_floor=0.0,
        frame_length=25,
        frame_shift=10,
        high_freq=0,
        low_freq=20,
        num_mel_bins=23,
        preemphasis_coefficient=0.97,
        raw_energy=True,
        remove_dc_offset=True,
        round_to_power_of_two=True,
        sample_frequency=16000,
        snip_edges=True,
        use_energy=False,
        use_log_fbank=True,
        use_power=True,
        window_type='povey'):
    """ (log) Mel filter banks

    :param waveform: Input waveform.
    :param blackman_coeff: Constant coefficient for generalized Blackman window. (float, default = 0.42)
    :param dither: Dithering constant (0.0 means no dither). If you turn this off, you should set the --energy-floor option, e.g. to 1.0 or 0.1 (float, default = 1)
    :param energy_floor: Floor on energy (absolute, not relative) in FBANK computation. Only makes a difference if --use-energy=true; only necessary if --dither=0.0.  Suggested values: 0.1 or 1.0 (float, default = 0)
    :param frame_length: Frame length in milliseconds (float, default = 25)
    :param frame_shift: Frame shift in milliseconds (float, default = 10)
    :param high_freq: High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)
    :param low_freq: Low cutoff frequency for mel bins (float, default = 20)
    :param num_mel_bins: Number of triangular mel-frequency bins (int, default = 23)
    :param preemphasis_coefficient: Coefficient for use in signal preemphasis (float, default = 0.97)
    :param raw_energy: If true, compute energy before preemphasis and windowing (bool, default = true)
    :param remove_dc_offset: Subtract mean from waveform on each frame (bool, default = true)
    :param round_to_power_of_two: If true, round window size to power of two by zero-padding input to FFT. (bool, default = true)
    :param sample_frequency: Waveform data sample frequency (must match the waveform file, if specified there) (float, default = 16000)
    :param snip_edges: If true, end effects will be handled by outputting only frames that completely fit in the file, and the number of frames depends on the frame-length.  If false, the number of frames depends only on the frame-shift, and we reflect the data at the ends. (bool, default = true)
    :param use_energy: Add an extra energy output. (bool, default = false)
    :param use_log_fbank: If true, produce log-filterbank, else produce linear. (bool, default = true)
    :param use_power: If true, use power, else use magnitude. (bool, default = true)
    :param window_type: Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"sine"|"blackmann") (string, default = "povey")
    :return: feat: (log) Mel filter banks.
    """
    window_size = int(frame_length * sample_frequency * 0.001)
    window_shift = int(frame_shift * sample_frequency * 0.001)
    frames, log_energy = extract_window(
        wavform=waveform,
        blackman_coeff=blackman_coeff,
        dither=dither,
        window_size=window_size,
        window_shift=window_shift,
        preemphasis_coefficient=preemphasis_coefficient,
        raw_energy=raw_energy,
        remove_dc_offset=remove_dc_offset,
        snip_edges=snip_edges,
        window_type=window_type
    )
    if round_to_power_of_two:
        n = 1
        while n < window_size:
            n *= 2
    else:
        n = window_size
    if use_power:
        spectrum = compute_power_spectrum(frames, n)
    else:
        spectrum = compute_spectrum(frames, n)
    mel_banks = get_mel_banks(
        num_bins=num_mel_bins,
        sample_frequency=sample_frequency,
        low_freq=low_freq,
        high_freq=high_freq,
        n=n
    )
    feat = np.dot(spectrum, mel_banks.T)
    if use_log_fbank:
        feat = np.log(feat.clip(min=np.finfo(float).eps))
    if use_energy:
        if energy_floor > 0.0:
            log_energy_floor = math.log(energy_floor)
            log_energy.clip(min=log_energy_floor)
        return feat, log_energy
    return feat


def compute_mfcc_feats(
        waveform,
        blackman_coeff=0.42,
        cepstral_lifter=22,
        dither=1.0,
        energy_floor=0.0,
        frame_length=25,
        frame_shift=10,
        high_freq=0,
        low_freq=20,
        num_ceps=13,
        num_mel_bins=23,
        preemphasis_coefficient=0.97,
        raw_energy=True,
        remove_dc_offset=True,
        round_to_power_of_two=True,
        sample_frequency=16000,
        snip_edges=True,
        use_energy=True,
        window_type='povey'):
    """ Mel-frequency cepstral coefficients

    :param waveform: Input waveform.
    :param blackman_coeff: Constant coefficient for generalized Blackman window. (float, default = 0.42)
    :param cepstral_lifter: Constant that controls scaling of MFCCs (float, default = 22)
    :param dither: Dithering constant (0.0 means no dither). If you turn this off, you should set the --energy-floor option, e.g. to 1.0 or 0.1 (float, default = 1)
    :param energy_floor: Floor on energy (absolute, not relative) in MFCC computation. Only makes a difference if --use-energy=true; only necessary if --dither=0.0.  Suggested values: 0.1 or 1.0 (float, default = 0)
    :param frame_length: Frame length in milliseconds (float, default = 25)
    :param frame_shift: Frame shift in milliseconds (float, default = 10)
    :param high_freq: High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)
    :param low_freq: Low cutoff frequency for mel bins (float, default = 20)
    :param num_ceps: Number of cepstra in MFCC computation (including C0) (int, default = 13)
    :param num_mel_bins: Number of triangular mel-frequency bins (int, default = 23)
    :param preemphasis_coefficient: Coefficient for use in signal preemphasis (float, default = 0.97)
    :param raw_energy: If true, compute energy before preemphasis and windowing (bool, default = true)
    :param remove_dc_offset: Subtract mean from waveform on each frame (bool, default = true)
    :param round_to_power_of_two: If true, round window size to power of two by zero-padding input to FFT. (bool, default = true)
    :param sample_frequency: Waveform data sample frequency (must match the waveform file, if specified there) (float, default = 16000)
    :param snip_edges: If true, end effects will be handled by outputting only frames that completely fit in the file, and the number of frames depends on the frame-length.  If false, the number of frames depends only on the frame-shift, and we reflect the data at the ends. (bool, default = true)
    :param use_energy: Use energy (not C0) in MFCC computation (bool, default = true)
    :param window_type: Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"sine"|"blackmann") (string, default = "povey")
    :return: feat: Mel-frequency cespstral coefficients.
    """
    feat, log_energy = compute_fbank_feats(
        waveform=waveform,
        blackman_coeff=blackman_coeff,
        dither=dither,
        energy_floor=energy_floor,
        frame_length=frame_length,
        frame_shift=frame_shift,
        high_freq=high_freq,
        low_freq=low_freq,
        num_mel_bins=num_mel_bins,
        preemphasis_coefficient=preemphasis_coefficient,
        raw_energy=raw_energy,
        remove_dc_offset=remove_dc_offset,
        round_to_power_of_two=round_to_power_of_two,
        sample_frequency=sample_frequency,
        snip_edges=snip_edges,
        use_energy=use_energy,
        use_log_fbank=True,
        use_power=True,
        window_type=window_type
    )
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :num_ceps]
    lifter_coeffs = compute_lifter_coeffs(cepstral_lifter, feat)
    feat = feat * lifter_coeffs
    if use_energy:
        feat[:, 0] = log_energy
    return feat


def apply_cmvn_sliding(feat, center=False, window=600, min_window=100, norm_vars=False):
    """ Sliding-window cepstral mean (and optionally variance) normalization

    :param feat: Cepstrum.
    :param center: If true, use a window centered on the current frame (to the extent possible, modulo end effects). If false, window is to the left. (bool, default = false)
    :param window: Window in frames for running average CMN computation (int, default = 600)
    :param min_window: Minimum CMN window used at start of decoding (adds latency only at start). Only applicable if center == false, ignored if center==true (int, default = 100)
    :param norm_vars: If true, normalize variance to one. (bool, default = false)
    :return: feat: Normalized cepstrum.
    """
    num_frames, feat_dim = feat.shape
    if center:
        mean1 = feat[:window].mean(axis=0, keepdims=True)
        if num_frames <= window:
            mean = mean1.repeat(num_frames, axis=0)
        else:
            mean2 = sliding_window(feat.T, window, 1).mean(axis=2).T
            mean3 = feat[-window:].mean(axis=0, keepdims=True)
            mean = np.concatenate([
                mean1.repeat(window // 2, axis=0),
                mean2,
                mean3.repeat((window - 1) // 2, axis=0)
            ])
    else:
        mean1 = feat[:min_window].mean(axis=0, keepdims=True)
        if num_frames <= min_window:
            mean = mean1.repeat(num_frames, axis=0)
        else:
            s = np.cumsum(feat[:window], axis=0)[min_window:]
            c = np.arange(min_window, min(window, num_frames), dtype=feat.dtype)[:, np.newaxis]
            mean2 = s / c
            if num_frames <= window:
                mean = np.concatenate([
                    mean1.repeat(min_window, axis=0),
                    mean2
                ])
            else:
                mean3 = sliding_window(feat.T, window, 1).mean(axis=2).T
                mean = np.concatenate([
                    mean1.repeat(min_window, axis=0),
                    mean2,
                    mean3[1:]
                ])
    feat = feat - mean
    return feat


def compute_spectrum(frames, n):
    complex_spec = np.fft.rfft(frames, n)
    return np.absolute(complex_spec)


def compute_power_spectrum(frames, n):
    return np.square(compute_spectrum(frames, n))


def mel_scale(hz):
    return 1127 * np.log(hz / 700.0 + 1)


def inverse_mel_scale(mel):
    return 700 * (np.exp(mel / 1127.0) - 1)


def get_mel_banks(num_bins, sample_frequency, low_freq, high_freq, n):
    assert num_bins >= 3, 'Must have at least 3 mel bins'
    num_fft_bins = n // 2

    nyquist = 0.5 * sample_frequency
    if high_freq <= 0:
        high_freq = nyquist + high_freq
    assert 0 <= low_freq < high_freq <= nyquist

    fft_bin_width = sample_frequency / n

    mel_low_freq = mel_scale(low_freq)
    mel_high_freq = mel_scale(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    mel_banks = np.zeros([num_bins, num_fft_bins + 1])
    for i in range(num_bins):
        left_mel = mel_low_freq + mel_freq_delta * i
        center_mel = left_mel + mel_freq_delta
        right_mel = center_mel + mel_freq_delta
        for j in range(num_fft_bins):
            mel = mel_scale(fft_bin_width * j)
            if left_mel < mel < right_mel:
                if mel <= center_mel:
                    mel_banks[i, j] = (mel - left_mel) / (center_mel - left_mel)
                else:
                    mel_banks[i, j] = (right_mel - mel) / (right_mel - center_mel)
    return mel_banks


def compute_lifter_coeffs(q, coeffs):
    return 1 + 0.5 * q * np.sin(np.pi * np.arange(coeffs.shape[1]) / q)
