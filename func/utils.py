import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Fungsi untuk menghitung resistivitas semu berdasarkan konfigurasi Schlumberger
def calculate_apparent_resistivity_schlumberger(ab, mn, voltage, current):
    geometric_factor = (np.pi * (ab**2 - mn**2)) / (2 * mn)
    apparent_resistivity = geometric_factor * (voltage / current)
    return apparent_resistivity


# Fungsi untuk menghitung resistivitas semu berdasarkan konfigurasi Schlumberger
def calculate_random(ab, resistivities, thicknesses):
    num_layers = len(resistivities)
    apparent_resistivities = np.zeros_like(ab)

    for i, a in enumerate(ab):
        # Koefisien refleksi lapisan terakhir
        reflection_coefficient = (resistivities[-1] - resistivities[-2]) / (
            resistivities[-1] + resistivities[-2]
        )

        # Hitung koefisien refleksi untuk lapisan lainnya
        for j in range(num_layers - 2, -1, -1):
            thickness = thicknesses[j]
            resistivity = resistivities[j]
            next_resistivity = resistivities[j + 1]
            k = 2 * np.pi / a  # Faktor gelombang
            r_factor = (next_resistivity - resistivity) / (
                next_resistivity + resistivity
            )
            reflection_coefficient = (
                r_factor + reflection_coefficient * np.exp(-2 * k * thickness)
            ) / (1 + r_factor * reflection_coefficient * np.exp(-2 * k * thickness))

        # Menghitung resistivitas semu untuk jarak AB/2
        apparent_resistivities[i] = (
            resistivities[0]
            * (1 - reflection_coefficient)
            / (1 + reflection_coefficient)
        )

    return apparent_resistivities


# Fungsi untuk menghasilkan dummy data geolistrik berdasarkan konfigurasi Schlumberger
def generate_data(num_layers=5, num_ab=20):
    # Membuat parameter resistivitas acak untuk setiap lapisan (10 - 500 Ohm.m)
    resistivities = np.random.uniform(low=0.1, high=2000, size=num_layers)

    # Membuat ketebalan acak untuk lapisan, kecuali lapisan terakhir (5 - 50 meter)
    thicknesses = np.random.uniform(low=2, high=30, size=num_layers - 1)

    # Jarak AB/2 dari 10 hingga 1000 meter (skala logaritmik)
    ab = np.logspace(1, 3, num_ab)

    # Hitung resistivitas semu berdasarkan konfigurasi Schlumberger
    apparent_resistivities = calculate_random(
        ab, resistivities, thicknesses
    )

    # Hasilkan data dalam format DataFrame
    data = pd.DataFrame({"AB": ab, "Apparent Resistivity": apparent_resistivities})

    return data


def plot_geology_cross_section(resistivities, thicknesses):
    # Filter hanya ketebalan yang valid (positif dan tidak None)
    valid_idx = (thicknesses >= 0) & (
        ~np.isnan(thicknesses)
    )  # Filter hanya thickness valid
    thicknesses_valid = np.array(thicknesses)[valid_idx]  # Ambil hanya thickness valid

    # Ambil resistivities yang sesuai dengan ketebalan valid
    resistivities_valid = np.array(resistivities)[: len(thicknesses_valid)]

    # Tambahkan ketebalan kumulatif untuk plotting
    depths = np.cumsum(np.insert(thicknesses_valid, 0, 0))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot setiap lapisan resistivitas yang valid
    for i in range(len(resistivities_valid)):
        ax.fill_between(
            [0, 1],
            depths[i],
            depths[i + 1],
            color=plt.cm.viridis(i / len(resistivities_valid)),
            label=f"Layer {i + 1}: {resistivities_valid[i]:.1f} Ohm.m",
        )

    ax.set_xlabel("Distance")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Geology Cross Section")
    ax.legend(loc="upper right")

    plt.gca().invert_yaxis()
    return fig


# Fungsi untuk menampilkan plot resistivitas semu vs AB/2 dan hasil inversi
def plot_inversion_results(data, ab, calculated_resistivity, depths, resistivities):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.scatter(ab, data["Apparent Resistivity"], color="black", label="Observed")
    ax1.plot(ab, calculated_resistivity, color="gray", label="Calculated")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("AB/2 (m)")
    ax1.set_ylabel("App. Resistivity (ohm.m)")
    ax1.set_title("Resistivity vs AB/2")
    ax1.legend()
    ax1.grid(True, which="both", ls="--")

    ax2.step(resistivities, depths, where="mid", color="black", label="Inversion Model")
    ax2.set_xscale("log")
    ax2.invert_yaxis()
    ax2.set_xlabel("Resistivity (ohm.m)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("1D Inversion Model")
    ax2.grid(True, which="both", ls="--")

    plt.tight_layout()
    return fig
