import opendssdirect as dss
import pandas as pd
import numpy as np
import random

# === CONFIG ===
# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

n_days = 7
hours_per_day = 24
total_hours = n_days * hours_per_day

# === LOAD IEEE 13-NODE MODEL ===
dss.Basic.ClearAll()
dss.run_command("Redirect data/raw/IEEE13Nodeckt.dss")

# === SOLAR PROFILE ===
daily_solar = [max(0, np.sin((np.pi / 12) * (h - 6))) for h in range(24)]

# === RANDOM FDIA HOURS & TYPES PER DAY ===
fdia_schedule = {}
for day in range(n_days):
    hours = sorted(np.random.choice(range(24), size=np.random.randint(2, 5), replace=False))
    fdia_schedule[day] = {
        h: random.choice([1, 2, 3, 4]) for h in hours
    }

# === SIMULATION ===
records = []
for t in range(total_hours):
    hour_of_day = t % 24
    day = t // 24
    is_fdia = hour_of_day in fdia_schedule.get(day, {})

    # Default PV
    true_pv_kw = daily_solar[hour_of_day] * 150

    # Apply PV falsification if needed
    if is_fdia and fdia_schedule[day][hour_of_day] == 3:
        pv_kw = true_pv_kw * random.uniform(0.5, 1.5)
    else:
        pv_kw = true_pv_kw

    # Set PV in simulator
    dss.Generators.Name("PV675")
    dss.Generators.kW(pv_kw)

    # Solve
    dss.Solution.Solve()

    # Read voltage magnitudes in pu (AllBusMagPu returns flattened list)
    buses = dss.Circuit.AllBusNames()
    vmag = dss.Circuit.AllBusMagPu()

    # Map bus names to voltage magnitudes (averaged by phase count)
    voltages = {}
    i = 0
    for bus in buses:
        num_nodes = len(dss.Bus.Nodes())
        dss.Circuit.SetActiveBus(bus)
        num_phases = dss.Bus.NumNodes()
        voltages[bus] = np.mean(vmag[i:i + num_phases]) if num_phases > 0 else 0
        i += num_phases

    row = voltages.copy()
    row["hour"] = hour_of_day
    row["day"] = day
    row["timestamp"] = f"day{day}_hour{hour_of_day}"
    row["pv_kw"] = pv_kw
    row["fdia"] = 1 if is_fdia else 0
    row["fdia_type"] = fdia_schedule[day][hour_of_day] if is_fdia else 0
    row["fdia_target_bus"] = None

    # Apply voltage manipulation only after reading base values
    bus_675_key = next((b for b in voltages if b.startswith("675")), None)
    if is_fdia and bus_675_key:
        if fdia_schedule[day][hour_of_day] == 1:
            row[bus_675_key] += np.random.uniform(0.1, 0.3)  # Voltage spike
        elif fdia_schedule[day][hour_of_day] == 2:
            row[bus_675_key] -= np.random.uniform(0.1, 0.3)  # Voltage drop
        elif fdia_schedule[day][hour_of_day] == 4:
            row[bus_675_key] += np.random.normal(0, 0.05)    # Voltage noise
        row["fdia_target_bus"] = bus_675_key

    records.append(row)

# === EXPORT ===
df = pd.DataFrame(records)
df.to_csv("data/processed/ieee13_multitype_fdia.csv", index=False)

# Show FDIA plan summary
print("FDIA schedule per day (hour: type):")
for day, plan in fdia_schedule.items():
    print(f"Day {day}: {plan}")
