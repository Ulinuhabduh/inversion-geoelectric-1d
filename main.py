import numpy as np
import pandas as pd
import streamlit as st
from func.models import (
    optimize_inversion,
    forward_model,
    create_inversion_dataframe,
    optimize_inversion_levenberg,
)
from func.utils import (
    generate_data,
    calculate_apparent_resistivity_schlumberger,
    plot_geology_cross_section,
    plot_inversion_results,
)

# Set page configuration
st.set_page_config(page_title="Geoelectric Inversion", layout="wide")

# Sidebar for page navigation
page = st.sidebar.selectbox("Select a page:", ["Inversion", "Dummy Data Generator"])

# Halaman 1: Inversi Geolistrik
if page == "Inversion":
    st.title("1D Geoelectric Inversion (Schlumberger Configuration)")
    st.write("Upload your geoelectric data for inversion and analysis.")

    # Sidebar inputs
    uploaded_file = st.sidebar.file_uploader("Upload your CSV data file", type=["csv"])
    st.sidebar.info(
        "Make sure the column name in the data is V, I, MN, AB or if there is already apparent resistivity then name the column 'Apparent Resistivity'."
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([2, 3])
        with col1:
            data = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data")
            st.dataframe(data.head())

        with col2:
            col1, col2 = st.columns([1, 1])

            with col1:
                # User selects optimization method and number of layers
                optimization_method = st.selectbox(
                    "Select Optimization Method",
                    [
                        "Nelder-Mead",
                        "BFGS",
                        "L-BFGS-B",
                        "Powell",
                        "SLSQP",
                        "COBYLA",
                        "Levenberg",
                    ],
                )
            with col2:
                num_layers = st.slider("Number of Layers", 2, 10, 4)

            # Add a start button to control when the inversion happens
            start_inversion = st.button("Start Inversion")

        if start_inversion: 
            progress = st.progress(0)  

            # Stage 1: Calculate Apparent Resistivity if necessary
            progress.progress(10)
            if "Apparent Resistivity" not in data.columns:
                st.sidebar.write("Apparent Resistivity is not in the data.")
                method = st.sidebar.selectbox(
                    "How do you want to calculate Apparent Resistivity?",
                    options=["Calculate from AB, MN, V, I", "Skip Calculation"],
                )

                if method == "Calculate from AB, MN, V, I":
                    ab = np.array(data["AB"])
                    mn = np.array(data["MN"])
                    voltage = np.array(data["V"])
                    current = np.array(data["I"])
                    data["Apparent Resistivity"] = (
                        calculate_apparent_resistivity_schlumberger(
                            ab, mn, voltage, current
                        )
                    )

            # Stage 2: Group Data
            progress.progress(30)
            data_grouped = data.groupby("AB").mean().reset_index()

            # Stage 3: Initial guesses for resistivity and thickness
            progress.progress(50)
            col1, col2 = st.columns([3, 2])

            with col1:
                initial_resistivities = np.ones(num_layers) * np.mean(
                    data_grouped["Apparent Resistivity"]
                )
                initial_thicknesses = np.linspace(0.1, 10, num_layers - 1)
                initial_guess = np.concatenate(
                    [initial_resistivities, initial_thicknesses]
                )

                # Stage 4: Optimization Process
                progress.progress(70)
                if optimization_method == "Levenberg":
                    result, error = optimize_inversion_levenberg(
                        data_grouped, initial_guess, num_layers
                    )
                else:
                    result, error = optimize_inversion(
                        data_grouped,
                        initial_guess,
                        num_layers,
                        optimization_method,
                    )

                # Stage 5: Calculate forward model for the inversion result
                progress.progress(85)
                calculated_resistivity = forward_model(
                    data_grouped["AB"], result, num_layers
                )

                # Stage 6: Plotting inversion results
                fig = plot_inversion_results(
                    data_grouped,
                    data_grouped["AB"],
                    calculated_resistivity,
                    np.cumsum(np.insert(result[num_layers:], 0, 0)),
                    result[:num_layers],
                )
                st.write("### Inversion Results")
                st.pyplot(fig)

                # Stage 7: Plotting geology cross-section
                st.write("### Geology Cross Section")
                thicknesses = result[num_layers:]  # Ambil ketebalan
                thicknesses = thicknesses[~np.isnan(thicknesses)]  # Hapus nilai NaN
                fig2 = plot_geology_cross_section(result[:num_layers], thicknesses)
                st.pyplot(fig2)

                # Finalize progress
                progress.progress(100)

                with col2:
                    # Buat DataFrame hasil inversi
                    inversion_df = create_inversion_dataframe(result, num_layers)

                    # Tampilkan DataFrame hasil inversi
                    st.write("#### Inversion Table")
                    st.dataframe(inversion_df)

                    # Tampilkan error inversi
                    st.error(f"### Inversion Error: {error/100:.2f}%")

    else:
        st.write("Please upload a CSV file to proceed.")

# Halaman 2: Dummy Data Generator
elif page == "Dummy Data Generator":
    st.title("Dummy Geoelectric Data Generator")
    st.write("Generate random geoelectric data for Schlumberger configuration.")

    num_layers = st.sidebar.slider("Number of Layers", 2, 10, 5)
    num_ab = st.sidebar.slider("Number of AB/2 Points", 10, 100, 30)

    if st.button("Generate Dummy Data"):
        dummy_data = generate_data(num_layers, num_ab)
        st.write("### Generated Dummy Data:")
        st.dataframe(dummy_data)

        csv = dummy_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="dummy_geolistrik_data.csv",
            mime="text/csv",
        )
