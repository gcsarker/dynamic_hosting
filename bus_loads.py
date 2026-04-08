import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from collections import defaultdict



def prepare_bus_loads(file_path, line_df, R, X, root_bus=114, seed=0, power_factor=0.9):
    """
    Prepare bus-level load profiles in per-unit and kW.

    Parameters
    ----------
    file_path : str
        Path to the parquet dataset containing building-level load data.
    line_df : pandas.DataFrame
        Must contain:
        - 'upstream bus'
        - 'downstream bus'
    R : np.ndarray
        Resistance matrix.
    X : np.ndarray
        Reactance matrix.
    root_bus : int, default=114
        Root/substation bus to exclude from load assignment.
    seed : int, default=0
        Random seed for assigning buildings to buses.
    power_factor : float, default=0.9
        Assumed power factor at each bus.

    Returns
    -------
    bus_load_pu : pandas.DataFrame
        Final bus load profile in per-unit after voltage and headroom scaling.
    bus_load_kw_final : pandas.DataFrame
        Final bus load profile in kW after voltage and headroom scaling.
    """

    dataset = ds.dataset(file_path, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()
    df = df[['timestamp', 'out.electricity.net.energy_consumption']]

    # Get all non-root buses
    buses = sorted(set(line_df["upstream bus"]).union(set(line_df["downstream bus"])))
    if root_bus in buses:
        buses.remove(root_bus)

    # Get building IDs from index level or column
    if "bldg_id" in df.columns:
        bldgs = df["bldg_id"].unique()
    else:
        bldgs = df.index.get_level_values("bldg_id").unique()

    # Assign each building to a random bus (excluding root bus)
    np.random.seed(seed)
    bus_assignment = pd.DataFrame({
        "bldg_id": bldgs,
        "bus": np.random.choice(buses, size=len(bldgs))
    })

    df.reset_index(inplace=True)
    df = df.merge(bus_assignment, on="bldg_id")

    # Aggregate loads at bus level
    bus_load = (
        df.groupby(["timestamp", "bus"])["out.electricity.net.energy_consumption"]
        .sum()
        .reset_index()
    )

    bus_load = bus_load.pivot(
        index="timestamp",
        columns="bus",
        values="out.electricity.net.energy_consumption"
    )

    bus_load = bus_load.fillna(0.0)
    bus_load = bus_load.reindex(columns=buses, fill_value=0.0)

    # Convert energy to kW: P = E / Δt, where Δt = 0.25 hr (15 minutes)
    bus_load_kw = bus_load / 0.25
    

    # Step 1: scale loads for voltage feasibility
    peak_load_kw_raw = bus_load_kw.sum(axis=1).max()
    S_base_temp = peak_load_kw_raw
    bus_load_pu_temp = bus_load_kw / S_base_temp

    pf = np.full(bus_load_pu_temp.shape[1], power_factor)
    eta_bus = np.sqrt(1.0 / (pf ** 2) - 1.0)
    Z = R + X @ np.diag(eta_bus)

    v_base = 1.0 - (bus_load_pu_temp.to_numpy() @ Z.T)
    vmin_actual = v_base.min()

    print("Raw peak load (kW):", peak_load_kw_raw)
    print("Minimum baseline voltage before scaling:", vmin_actual)


    scale_voltage = (1.0 - 0.95) / (1.0 - vmin_actual)
    bus_load_kw_scaled = bus_load_kw * scale_voltage
    # print(bus_load_kw_scaled.head())


    # Step 2: transformer capacity after voltage scaling
    peak_load_kw_scaled = bus_load_kw_scaled.sum(axis=1).max()
    S_base = peak_load_kw_scaled
    bus_load_pu = bus_load_kw_scaled / S_base

    v_base = 1.0 - (bus_load_pu.to_numpy() @ Z.T)
    vmin_actual = v_base.min()

    print("\n")
    print("Applied voltage scaling factor:", scale_voltage)
    print("Peak load after voltage scaling (kW):", peak_load_kw_scaled)
    print("Transformer rating S_base (kW):", S_base)
    print("Peak load in pu after voltage scaling:", bus_load_pu.sum(axis=1).max())
    print("Minimum baseline voltage after scaling:", vmin_actual)

    # Step 3: additional scaling to create headroom
    bus_load_pu = 0.9 * bus_load_pu
    bus_load_kw_final = 0.9 * bus_load_kw_scaled

    # print(bus_load_kw_final.head())

    v_base = 1.0 - (bus_load_pu.to_numpy() @ Z.T)
    vmin_final = v_base.min()

    print("\n")
    print("Final peak load after headroom scaling (kW):", bus_load_kw_final.sum(axis=1).max())
    print("Transformer rating remains fixed (kW):", S_base)
    print("Final peak load in pu:", bus_load_pu.sum(axis=1).max())
    print("Minimum voltage after headroom scaling:", vmin_final)
    print("Available headroom at peak (pu):", 1.0 - bus_load_pu.sum(axis=1).max())
    print("Available headroom at peak (kW):", S_base - bus_load_kw_final.sum(axis=1).max())

    return S_base, bus_load_pu, bus_load_kw_final