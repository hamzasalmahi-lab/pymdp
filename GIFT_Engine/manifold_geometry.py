import numpy as np

def calculate_k_score(belief_vector):
    # The GIFT-Score (k)
    # If the agent is 100% sure, k is 1.0 (Deep Well).
    # If the agent is 50/50 confused, k is 0.0 (Flat Snap).
    k = np.abs(belief_vector[0] - 0.5) * 2
    return k
