import numpy as np


def sliding_window(x, window_size, window_shift):
    shape = x.shape[:-1] + (x.shape[-1] - window_size + 1, window_size)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)[::window_shift]


def func_num_frames(num_samples, window_size, window_shift, snip_edges):
    if snip_edges:
        if num_samples < window_size:
            return 0
        else:
            return 1 + ((num_samples - window_size) // window_shift)
    else:
        return (num_samples + (window_shift // 2)) // window_shift


def func_dither(waveform, dither_value):
    if dither_value == 0.0:
        return waveform
    waveform += np.random.normal(size=waveform.shape) * dither_value
    return waveform


def func_remove_dc_offset(waveform):
    return waveform - np.mean(waveform)


def func_log_energy(waveform):
    return np.log(np.dot(waveform, waveform).clip(min=np.finfo(float).eps))


def func_preemphasis(waveform, preemph_coeff):
    if preemph_coeff == 0.0:
        return waveform
    assert 0 <= preemph_coeff <= 1
    waveform[1:] -= preemph_coeff * waveform[:-1]
    waveform[0] -= preemph_coeff * waveform[0]
    return waveform


def sine(M):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, float)
    n = np.arange(0, M)
    return np.sin(np.pi*n/(M-1))


def povey(M):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, float)
    n = np.arange(0, M)
    return (0.5 - 0.5*np.cos(2.0*np.pi*n/(M-1)))**0.85


def feature_window_function(window_type, window_size, blackman_coeff=0.42):
    assert window_size > 0
    if window_type == 'hanning':
        return np.hanning(window_size)
    elif window_type == 'sine':
        return sine(window_size)
    elif window_type == 'hamming':
        return np.hamming(window_size)
    elif window_type == 'povey':
        return povey(window_size)
    elif window_type == 'rectangular':
        return np.ones(window_size)
    elif window_type == 'blackman':
        return np.blackman(window_size) - 0.42 + blackman_coeff
    else:
        raise ValueError('Invalid window type {}'.format(window_type))


def process_window(window, dither, remove_dc_offset, preemphasis_coefficient, window_function, raw_energy):
    if dither != 0.0:
        window = func_dither(window, dither)
    if remove_dc_offset:
        window = func_remove_dc_offset(window)
    if raw_energy:
        log_energy = func_log_energy(window)
    if preemphasis_coefficient != 0.0:
        window = func_preemphasis(window, preemphasis_coefficient)
    window *= window_function
    if not raw_energy:
        log_energy = func_log_energy(window)
    return window, log_energy


def extract_window(wavform, blackman_coeff, dither, window_size, window_shift,
                   preemphasis_coefficient, raw_energy, remove_dc_offset,
                   snip_edges, window_type):
    num_samples = len(wavform)
    num_frames = func_num_frames(num_samples, window_size, window_shift, snip_edges)
    num_samples_ = (num_frames - 1) * window_shift + window_size
    if snip_edges:
        wavform = wavform[:num_samples_]
    else:
        offset = window_shift // 2 - window_size // 2
        wavform = np.concatenate([
            wavform[-offset - 1::-1],
            wavform,
            wavform[:-(offset + num_samples_ - num_samples + 1):-1]
        ])
    frames = sliding_window(wavform, window_size=window_size, window_shift=window_shift).astype(np.float32)
    log_enery = np.empty(frames.shape[0], dtype=frames.dtype)
    for i in range(frames.shape[0]):
        frames[i], log_enery[i] = process_window(
            window=frames[i],
            dither=dither,
            remove_dc_offset=remove_dc_offset,
            preemphasis_coefficient=preemphasis_coefficient,
            window_function=feature_window_function(window_type, window_size, blackman_coeff),
            raw_energy=raw_energy
        )
    return frames, log_enery
