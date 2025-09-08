import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

st.title("Demographic Modeling App (WPP 2024)")

# Load data from output.csv
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("output.csv", skiprows=14)  # Skip metadata
        df.columns = [
            'Index', 'Variant', 'Location', 'Notes', 'Location code', 'ISO3 Alpha-code', 'ISO2 Alpha-code',
            'SDMX code', 'Type', 'Parent code', 'Year', 'TPopulation1Jan', 'TPopulation1July',
            'MalePop1July', 'FemalePop1July', 'PopDensity', 'SexRatio', 'MedianAge',
            'NaturalChange', 'RateNaturalChange', 'PopChange', 'PopGrowthRate', 'DoublingTime',
            'Births', 'TeenBirths', 'CrudeBirthRate', 'TFR', 'NetReproductionRate', 'MAC',
            'SexRatioBirth', 'TotalDeaths', 'MaleDeaths', 'FemaleDeaths', 'CrudeDeathRate',
            'LEx', 'MaleLEx', 'FemaleLEx', 'LEx15', 'MaleLEx15', 'FemaleLEx15',
            'LEx65', 'MaleLEx65', 'FemaleLEx65', 'LEx80', 'MaleLEx80', 'FemaleLEx80',
            'InfantDeaths', 'InfantMortalityRate', 'LiveBirthsSurviving1', 'UnderFiveDeaths',
            'UnderFiveMortality', 'MortalityBefore40', 'MaleMortalityBefore40', 'FemaleMortalityBefore40',
            'MortalityBefore60', 'MaleMortalityBefore60', 'FemaleMortalityBefore60',
            'Mortality15to50', 'MaleMortality15to50', 'FemaleMortality15to50',
            'Mortality15to60', 'MaleMortality15to60', 'FemaleMortality15to60',
            'NetMigrants', 'NetMigrationRate'
        ]
        numeric_cols = [
            'Year', 'TPopulation1Jan', 'TPopulation1July', 'MalePop1July', 'FemalePop1July',
            'PopDensity', 'SexRatio', 'MedianAge', 'NaturalChange', 'RateNaturalChange',
            'PopChange', 'PopGrowthRate', 'DoublingTime', 'Births', 'TeenBirths',
            'CrudeBirthRate', 'TFR', 'NetReproductionRate', 'MAC', 'SexRatioBirth',
            'TotalDeaths', 'MaleDeaths', 'FemaleDeaths', 'CrudeDeathRate', 'LEx',
            'MaleLEx', 'FemaleLEx', 'LEx15', 'MaleLEx15', 'FemaleLEx15', 'LEx65',
            'MaleLEx65', 'FemaleLEx65', 'LEx80', 'MaleLEx80', 'FemaleLEx80',
            'InfantDeaths', 'InfantMortalityRate', 'LiveBirthsSurviving1', 'UnderFiveDeaths',
            'UnderFiveMortality', 'MortalityBefore40', 'MaleMortalityBefore40', 'FemaleMortalityBefore40',
            'MortalityBefore60', 'MaleMortalityBefore60', 'FemaleMortalityBefore60',
            'Mortality15to50', 'MaleMortality15to50', 'FemaleMortality15to50',
            'Mortality15to60', 'MaleMortality15to60', 'FemaleMortality15to60',
            'NetMigrants', 'NetMigrationRate'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Create sortable table for TFR
st.subheader("Fertility Rates by Location (2023)")
tfr_data = df[(df['Year'] == 2023) & (df['Variant'] == 'Estimates') & ((df['Type'] == 'Country/Area') | (df['Type'] == 'World'))][['Location', 'TFR']]
tfr_data = tfr_data.sort_values(by='TFR', ascending=True).reset_index(drop=True)
st.dataframe(tfr_data, use_container_width=True, column_config={
    "Location": "Location",
    "TFR": st.column_config.NumberColumn("Total Fertility Rate", format="%.2f")
})

# Filter for countries and world
locations = sorted(df[(df['Type'] == 'Country/Area') | (df['Type'] == 'World')]['Location'].unique())

location = st.selectbox("Select Location", locations)

# Use most recent year (2023) for initial conditions
current_year = 2023
location_df = df[(df['Location'] == location) & (df['Year'] == current_year) & (df['Variant'] == 'Estimates')]

if not location_df.empty:
    pop = location_df['TPopulation1Jan'].values[0] * 1000  # Convert thousands to absolute
    tfr = location_df['TFR'].values[0]
    mac = location_df['MAC'].values[0]  # Mean age at childbearing
    lex = location_df['LEx'].values[0]  # Life expectancy at birth
    migration_rate = location_df['NetMigrationRate'].values[0] / 1000  # Convert per 1,000 to fraction

    st.subheader(f"Current Data ({current_year})")
    st.write(f"Population: {pop:,}")
    st.write(f"Total Fertility Rate (TFR): {tfr:.2f}")
    st.write(f"Mean Age at Childbearing: {mac:.1f} years")
    st.write(f"Life Expectancy: {lex:.1f} years")
    st.write(f"Net Migration Rate: {migration_rate*1000:.2f} per 1,000")

    st.subheader("Scenario Parameters")
    scenario = st.selectbox("Scenario", ["Base", "Low Fertility", "High Fertility", "Catastrophic", "Custom"])

    if scenario == "Base":
        adj_tfr = tfr
        adj_mac = mac
        adj_pop = pop
        adj_replacement = 2.1
        adj_migration = migration_rate
    elif scenario == "Low Fertility":
        adj_tfr = max(tfr - 0.5, 0.5)
        adj_mac = mac
        adj_pop = pop
        adj_replacement = 2.1
        adj_migration = migration_rate
    elif scenario == "High Fertility":
        adj_tfr = tfr + 0.5
        adj_mac = mac
        adj_pop = pop
        adj_replacement = 2.1
        adj_migration = migration_rate
    elif scenario == "Catastrophic":
        adj_tfr = 0.8
        adj_mac = mac
        adj_pop = pop * 0.5  # 50% immediate loss
        adj_replacement = 2.3  # Assume lower life expectancy
        adj_migration = migration_rate * 0.5  # Reduced migration
    else:  # Custom
        adj_tfr = st.slider("Adjusted TFR", 0.0, 5.0, float(tfr), format="%.2f")
        adj_mac = st.slider("Generation Time (years)", 20.0, 40.0, float(mac), format="%.1f")
        adj_pop = pop * st.slider("Initial Population Adjustment Factor", 0.1, 2.0, 1.0)
        adj_replacement = st.slider("Replacement TFR", 2.0, 2.5, 2.1)
        adj_migration = st.slider("Adjusted Net Migration Rate (per 1,000)", -10.0, 10.0, float(migration_rate*1000)) / 1000

    years_ahead = st.slider("Projection Years", 50, 200, 100)

    # Model type selection, default to Stochastic Itô for global projections
    model_type = st.selectbox("Model Type", ["Stochastic Itô (GBM)", "Deterministic Exponential", "Galton-Watson Branching"], index=1 if location == 'World' else 0)

    # Calculate base growth rate
    R0 = adj_tfr / adj_replacement
    r = math.log(R0) / adj_mac if R0 > 0 else -0.1
    r += adj_migration  # Incorporate migration

    # Projection setup
    dt = 1  # Time step in years
    t = np.arange(0, years_ahead + 1, dt)
    years = current_year + t

    if model_type == "Deterministic Exponential":
        pops = adj_pop * np.exp(r * t)
        proj_df = pd.DataFrame({"Year": years, "Population": pops})
        st.line_chart(proj_df.set_index("Year"))

        fig, ax = plt.subplots()
        ax.plot(years, pops / 1e9, label="Projected Population (Billions)", color='blue')
        ax.set_xlabel("Year")
        ax.set_ylabel("Population (Billions)")
        ax.set_title(f"Deterministic Projection for {location} under {scenario}")
        ax.legend()
        st.pyplot(fig)

    elif model_type == "Stochastic Itô (GBM)":
        # Add TFR variance slider
        tfr_std = st.slider("TFR Standard Deviation", 0.0, 2.0, 0.5 if location == 'World' else 1.2, format="%.2f")
        # Convert TFR standard deviation to volatility
        sigma = tfr_std / (adj_tfr * adj_mac) if adj_tfr > 0 and adj_mac > 0 else 0.02
        sigma = max(sigma, 0.02)  # Ensure minimum volatility
        st.write(f"Calculated Volatility (sigma): {sigma:.4f} per year")
        num_paths = st.slider("Number of Simulation Paths", 10, 500, 100)

        paths = np.zeros((len(t), num_paths))
        paths[0] = adj_pop
        for i in range(1, len(t)):
            dW = np.random.normal(0, np.sqrt(dt), num_paths)
            paths[i] = paths[i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)

        mean_pop = np.mean(paths, axis=1)
        std_pop = np.std(paths, axis=1)

        fig, ax = plt.subplots()
        ax.plot(years, mean_pop / 1e9, label="Mean Population (Billions)", color='blue')
        ax.fill_between(years, (mean_pop - std_pop) / 1e9, (mean_pop + std_pop) / 1e9, alpha=0.2, color='blue', label="1 Std Dev")
        ax.set_xlabel("Year")
        ax.set_ylabel("Population (Billions)")
        ax.set_title(f"Stochastic Itô (GBM) Projection for {location} under {scenario}")
        ax.legend()
        st.pyplot(fig)

    elif model_type == "Galton-Watson Branching":
        m = R0  # Mean offspring per individual
        num_generations = int(years_ahead / adj_mac) + 1
        gen_years = np.arange(0, num_generations) * adj_mac + current_year
        num_sims = st.slider("Number of Simulations", 10, 500, 100)

        gen_pops = np.zeros((num_generations, num_sims))
        gen_pops[0] = adj_pop
        variance_per_gen = m  # Poisson variance assumption

        for g in range(1, num_generations):
            gen_pops[g] = np.random.normal(gen_pops[g-1] * m, np.sqrt(gen_pops[g-1] * m), num_sims)
            gen_pops[g] = np.maximum(gen_pops[g], 0)  # Prevent negative populations

        mean_pop = np.mean(gen_pops, axis=1)
        std_pop = np.std(gen_pops, axis=1)

        fig, ax = plt.subplots()
        ax.plot(gen_years, mean_pop / 1e9, label="Mean Population (Billions)", color='blue')
        ax.fill_between(gen_years, (mean_pop - std_pop) / 1e9, (mean_pop + std_pop) / 1e9, alpha=0.2, color='blue', label="1 Std Dev")
        ax.set_xlabel("Year (Generations)")
        ax.set_ylabel("Population (Billions)")
        ax.set_title(f"Galton-Watson Branching Projection for {location} under {scenario}")
        ax.legend()
        st.pyplot(fig)

    # Time to collapse if declining
    if r < 0 and model_type == "Deterministic Exponential":
        time_to_one = -math.log(adj_pop) / r
        st.write(f"Estimated time to population collapse (to ~1 person): {time_to_one:.0f} years")
    elif r < 0:
        st.write("Stochastic model: Time to collapse varies due to randomness.")
    else:
        st.write("Population is growing or stable; no collapse projected.")

else:
    st.error(f"No data available for {location} in {current_year}.")

st.info("Data source: UN World Population Prospects 2024. Stochastic models use approximations; real-world dynamics may include additional factors (e.g., policy changes, catastrophes). For global projections, select 'World' and use the Stochastic Itô model.")
