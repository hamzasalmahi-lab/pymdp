import numpy as np

def get_observation_from_eeg(raw_data_chunk):
    # Imagine this looks at Alpha waves. 
    # If the wave is steady, it returns 0 (Stable).
    # If the wave is messy, it returns 1 (Unstable).
    variance = np.var(raw_data_chunk)
    return 0 if variance < 0.5 else 1
