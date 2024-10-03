import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

# Fungsi untuk menghitung error inversi
def inversion_error(params, ab, data_apparent_resistivity, num_layers):
    calculated_resistivity = forward_model(ab, params, num_layers)
    error = np.sum((data_apparent_resistivity - calculated_resistivity) ** 2)
    return error


# Inversi data untuk menemukan error terkecil
def optimize_inversion(data, initial_guess, num_layers, optimization_method):
    ab = data["AB"]
    apparent_resistivity = data["Apparent Resistivity"]

    result = minimize(
        inversion_error,
        initial_guess,
        args=(ab, apparent_resistivity, num_layers),
        method= optimization_method,
    )
    return result.x, result.fun

from scipy.optimize import least_squares

# Fungsi inversi menggunakan Levenberg-Marquardt (Damped Least Squares)
def optimize_inversion_levenberg(data, initial_guess, num_layers):
    ab = data["AB"]
    apparent_resistivity = data["Apparent Resistivity"]

    # Fungsi untuk menghitung residu antara resistivitas teramati dan terhitung
    def residuals(params):
        resistivities = params[:num_layers]
        thicknesses = params[num_layers:]
        calculated_resistivity = forward_model(ab, params, num_layers)
        return apparent_resistivity - calculated_resistivity

    result = least_squares(residuals, initial_guess, method='lm')  # Menggunakan metode Levenberg-Marquardt
    return result.x, result.cost


# Fungsi untuk menghitung resistivitas semu berdasarkan konfigurasi Schlumberger
def forward_model(ab, params, num_layers):
    resistivities = params[:num_layers]
    thicknesses = params[num_layers:]

    num_layers = len(resistivities)
    apparent_resistivities = np.zeros_like(ab)

    for i, a in enumerate(ab):
        reflection_coefficient = (resistivities[-1] - resistivities[-2]) / (
            resistivities[-1] + resistivities[-2]
        )
        for j in range(num_layers - 2, -1, -1):
            thickness = thicknesses[j]
            resistivity = resistivities[j]
            next_resistivity = resistivities[j + 1]
            k = 2 * np.pi / a
            r_factor = (next_resistivity - resistivity) / (
                next_resistivity + resistivity
            )
            reflection_coefficient = (
                r_factor + reflection_coefficient * np.exp(-2 * k * thickness)
            ) / (1 + r_factor * reflection_coefficient * np.exp(-2 * k * thickness))

        apparent_resistivities[i] = (
            resistivities[0]
            * (1 - reflection_coefficient)
            / (1 + reflection_coefficient)
        )

    return apparent_resistivities


# Fungsi untuk menampilkan hasil inversi dalam bentuk DataFrame
def create_inversion_dataframe(result, num_layers):
    resistivities = result[:num_layers]
    thicknesses = result[num_layers:]
    depths = np.cumsum(np.insert(thicknesses, 0, 0))

    data = {
        "Layer": [i + 1 for i in range(num_layers)],
        "Resistivity (Ohm.m)": resistivities,
        "Thickness (m)": np.append(thicknesses, np.nan),
        "Depth (m)": depths,
    }
    inversion_df = pd.DataFrame(data)
    
    return inversion_df
