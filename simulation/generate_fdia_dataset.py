import opendssdirect as dss
import pandas as pd
import numpy as np
import random

# === CONFIG ===
# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

n_days = 300
hours_per_day = 24
total_hours = n_days * hours_per_day

print(f"🔄 Initializing FDIA dataset generation...")
print(f"📅 Simulation period: {n_days} days ({total_hours} hours)")

# Initialize OpenDSS
dss.Basic.ClearAll()
try:
    dss.Command(r"Redirect data/raw/IEEE13Nodeckt.dss")
    print("✅ IEEE 13-bus system loaded successfully")
except Exception as e:
    print(f"❌ Failed to load IEEE 13-bus system: {e}")
    raise

# Verify system loaded correctly
num_buses = dss.Circuit.NumBuses()
print(f"📊 System has {num_buses} buses")

if num_buses == 0:
    raise ValueError("No buses found - check IEEE13Nodeckt.dss file path")

# === SOLAR PROFILE ===
daily_solar = [max(0, np.sin((np.pi / 12) * (h - 6))) for h in range(24)]

# === RANDOM FDIA HOURS & TYPES PER DAY ===
print("🎲 Generating FDIA attack schedule...")

fdia_schedule = {}
pv_buses = ["675", "692", "633"]
fdia_types = [1, 2, 3, 4]  # All attack types

total_fdia_hours = 0
type_counts = {1: 0, 2: 0, 3: 0, 4: 0}

for day in range(n_days):
    # Each day has 2-4 FDIA hours randomly distributed
    num_attacks = np.random.randint(2, 5)
    hours = sorted(np.random.choice(range(24), size=num_attacks, replace=False))
    
    fdia_schedule[day] = {}
    for h in hours:
        attack_type = random.choice(fdia_types)
        target_bus = random.choice(pv_buses)
        
        fdia_schedule[day][h] = {
            "type": attack_type,
            "target": target_bus
        }
        
        total_fdia_hours += 1
        type_counts[attack_type] += 1

print(f"📊 FDIA Schedule Summary:")
print(f"   Total attack hours: {total_fdia_hours}/{total_hours} ({total_fdia_hours/total_hours*100:.1f}%)")
print(f"   Type 1 (Voltage spike): {type_counts[1]} hours")
print(f"   Type 2 (Voltage drop): {type_counts[2]} hours") 
print(f"   Type 3 (PV manipulation): {type_counts[3]} hours")
print(f"   Type 4 (Voltage noise): {type_counts[4]} hours")

# === SIMULATION ===
records = []
for t in range(total_hours):
    hour_of_day = t % 24
    day = t // 24
    fdia_info = fdia_schedule.get(day, {}).get(hour_of_day)
    is_fdia = fdia_info is not None
    fdia_type = fdia_info["type"] if is_fdia else 0
    fdia_target_bus = fdia_info["target"] if is_fdia else None

    # Total PV injection this hour
    total_pv_kw = 0

    # Define PV buses and their max capacity
    pv_config = {
        "675": 50,
        "692": 50,  
        "633": 50
    }

    for bus, max_kw in pv_config.items():
        # Calculate base pv_power for each bus
        pv_power = daily_solar[hour_of_day] * max_kw
        
        # Apply FDIA Type 3 (PV manipulation) if this bus is targeted
        if is_fdia and fdia_type == 3 and fdia_target_bus == bus:
            pv_power *= random.uniform(0.5, 1.5)  # 50% to 150% of normal

        try:
            dss.Command(f"Generator.PV{bus}.kW={pv_power}")
        except Exception as e:
            print(f"Warning: Failed to set PV{bus} power: {e}")

        total_pv_kw += pv_power

    # Solve power flow
    try:
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            print(f"Warning: Power flow did not converge at hour {t}")
    except Exception as e:
        print(f"Error solving power flow at hour {t}: {e}")
        continue

    # Get bus voltages
    buses = dss.Circuit.AllBusNames()
    vmag = dss.Circuit.AllBusMagPu()

    voltages = {}
    i = 0
    for bus in buses:
        dss.Circuit.SetActiveBus(bus)
        num_phases = dss.Bus.NumNodes()
        voltages[bus] = np.mean(vmag[i:i + num_phases]) if num_phases > 0 else 0
        i += num_phases

    # Create record
    row = voltages.copy()
    row["hour"] = hour_of_day
    row["day"] = day
    row["timestamp"] = f"day{day}_hour{hour_of_day}"
    row["pv_kw"] = total_pv_kw
    row["fdia"] = 1 if is_fdia else 0
    row["fdia_type"] = fdia_type
    row["fdia_target_bus"] = fdia_target_bus

    # Apply FDIA voltage manipulations (Types 1, 2, 4)
    if is_fdia and fdia_target_bus:
        target_bus_key = next((b for b in voltages if b.startswith(fdia_target_bus)), None)
        if target_bus_key:
            if fdia_type == 1:  # Voltage spike attack
                spike = np.random.uniform(0.05, 0.15)  # 5-15% voltage increase
                row[target_bus_key] += spike
                row[target_bus_key] = min(row[target_bus_key], 1.5)  # Cap at 150%
            elif fdia_type == 2:  # Voltage drop attack
                drop = np.random.uniform(0.05, 0.15)  # 5-15% voltage decrease
                row[target_bus_key] -= drop
                row[target_bus_key] = max(row[target_bus_key], 0.8)  # Floor at 80%
            elif fdia_type == 4:  # Voltage noise injection
                noise = np.random.normal(0, 0.02)  # 2% standard deviation noise
                row[target_bus_key] += noise

    records.append(row)
    
    # Progress indicator
    if t % 240 == 0:  # Every 10 days
        progress = (t / total_hours) * 100
        print(f"Progress: {progress:.1f}% ({t}/{total_hours} hours)")

print("Simulation completed!")
df = pd.DataFrame(records)

# Add some statistics
fdia_counts = df.groupby('fdia_type').size()
print(f"\n📊 FDIA Statistics:")
print(f"Normal hours: {fdia_counts.get(0, 0)}")
print(f"Type 1 (Voltage spike): {fdia_counts.get(1, 0)}")
print(f"Type 2 (Voltage drop): {fdia_counts.get(2, 0)}")
print(f"Type 3 (PV manipulation): {fdia_counts.get(3, 0)}")
print(f"Type 4 (Voltage noise): {fdia_counts.get(4, 0)}")
print(f"Total FDIA hours: {df['fdia'].sum()}")
print(f"FDIA percentage: {(df['fdia'].sum() / len(df) * 100):.1f}%")

# Save data
df.to_csv("data/processed/ieee13_multitype_fdia.csv", index=False)
print(f"\n✅ Dataset saved: {len(df)} records to ieee13_multitype_fdia.csv")
