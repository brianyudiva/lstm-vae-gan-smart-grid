import opendssdirect as dss
import pandas as pd
import numpy as np
import random
import os
import json
from datetime import datetime, timedelta

# === CONFIG ===
np.random.seed(42)
random.seed(42)

CONFIG = {
    'n_days': 300,
    'hours_per_day': 24,
    'fdia_rate': 0.15,  # 15% of hours have attacks
    'min_attacks_per_day': 1,
    'max_attacks_per_day': 4,
    'convergence_tolerance': 1e-6,
    'max_retries': 3,
    'output_dir': 'data/processed',
}

total_hours = CONFIG['n_days'] * CONFIG['hours_per_day']

print(f"Initializing FDIA dataset generation")
print(f"Simulation period: {CONFIG['n_days']} days ({total_hours} hours)")
print(f"Target FDIA rate: {CONFIG['fdia_rate']*100:.1f}%")

def create_realistic_profiles():
    """Create realistic solar and load profiles"""
    daily_solar = []
    for h in range(24):
        base_solar = max(0, np.sin((np.pi / 12) * (h - 6)))
        noise = np.random.normal(0, 0.1)
        solar = max(0, min(1, base_solar + noise))
        daily_solar.append(solar)
    
    daily_load = []
    for h in range(24):
        if 6 <= h <= 8 or 17 <= h <= 22:  # Peak hours
            load_factor = 0.9 + np.random.normal(0, 0.05)
        elif 22 <= h or h <= 6:  # Night hours
            load_factor = 0.4 + np.random.normal(0, 0.03)
        else:  # Off-peak
            load_factor = 0.6 + np.random.normal(0, 0.04)
        
        daily_load.append(max(0.3, min(1.0, load_factor)))
    
    return daily_solar, daily_load

class FDIAAttackGenerator:
    def __init__(self):
        self.pv_buses = ["675", "692", "633"]
        self.critical_buses = ["632", "633", "634", "645", "646"]

        self.attack_types = {
            1: {"name": "Voltage Spike", "severity": "high", "detection_difficulty": "medium"},
            2: {"name": "Voltage Sag", "severity": "high", "detection_difficulty": "medium"}, 
            3: {"name": "PV Manipulation", "severity": "medium", "detection_difficulty": "hard"},
            4: {"name": "Measurement Noise", "severity": "low", "detection_difficulty": "hard"},
            5: {"name": "Coordinated Attack", "severity": "critical", "detection_difficulty": "medium"},
            6: {"name": "Stealthy Bias", "severity": "medium", "detection_difficulty": "very_hard"}
        }

    def generate_attack_schedule(self, n_days):
        """Generate attack schedule"""
        schedule = {}
        total_attacks = 0
        type_counts = {i: 0 for i in range(1, 7)}
        
        for day in range(n_days):
            # Vary attack frequency (some days no attacks, some days multiple)
            attack_prob = np.random.random()
            if attack_prob < 0.3:  # 30% of days have no attacks
                continue
            elif attack_prob < 0.7:  # 40% have 1-2 attacks
                num_attacks = np.random.randint(1, 3)
            else:  # 30% have 2-4 attacks
                num_attacks = np.random.randint(2, 5)
            
            # Prefer attacks during peak hours when impact is higher
            peak_hours = list(range(7, 10)) + list(range(17, 22))
            off_peak_hours = list(range(0, 7)) + list(range(10, 17)) + list(range(22, 24))
            
            hours = []
            for _ in range(num_attacks):
                if np.random.random() < 0.7:  # 70% during peak
                    hour = np.random.choice(peak_hours)
                else:
                    hour = np.random.choice(off_peak_hours)
                
                if hour not in hours:  # Avoid duplicate hours
                    hours.append(hour)
            
            schedule[day] = {}
            for h in hours:
                attack_type = self._select_attack_type()
                target_bus = self._select_target_bus(attack_type)
                
                schedule[day][h] = {
                    "type": attack_type,
                    "target": target_bus,
                    "severity": np.random.uniform(0.5, 1.0)  # Attack intensity
                }
                
                total_attacks += 1
                type_counts[attack_type] += 1
        
        return schedule, total_attacks, type_counts
    
    def _select_attack_type(self):
        """Select attack type with probabilities"""
        # More sophisticated attacks are less frequent
        probabilities = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]  # Types 1-6
        return np.random.choice(range(1, 7), p=probabilities)
    
    def _select_target_bus(self, attack_type):
        """Select target bus based on attack type"""
        if attack_type == 3:  # PV manipulation
            return np.random.choice(self.pv_buses)
        elif attack_type in [5, 6]:  # Coordinated/stealthy attacks
            return np.random.choice(self.critical_buses)
        else:
            all_buses = self.pv_buses + self.critical_buses
            return np.random.choice(all_buses)
        
def simulate_with_fdia(hour_of_day, day, fdia_info, daily_solar, daily_load):
    max_retries = CONFIG['max_retries']
    
    for attempt in range(max_retries):
            # Apply load profile
            total_load_kw = sum(daily_load) * 100  # Scale appropriately
            
            # Configure PV systems
            total_pv_kw = 0
            pv_config = {"675": 50, "692": 50, "633": 50}
            
            for bus, max_kw in pv_config.items():
                pv_power = daily_solar[hour_of_day] * max_kw
                
                # Apply FDIA Type 3 (PV manipulation)
                if (fdia_info and fdia_info["type"] == 3 and 
                    fdia_info["target"] == bus):
                    severity = fdia_info.get("severity", 0.5)
                    multiplier = 0.5 + severity * 0.5  # 0.5 to 1.0 range
                    pv_power *= np.random.uniform(multiplier, 1.5)
                
                dss.Command(f"Generator.PV{bus}.kW={pv_power}")
                total_pv_kw += pv_power
            
            # Solve power flow
            dss.Solution.Solve()
            
            if not dss.Solution.Converged():
                if attempt < max_retries - 1:
                    print(f"⚠️ Convergence failed at hour {day*24 + hour_of_day}, attempt {attempt + 1}")
                    continue
                else:
                    print(f"❌ Final convergence failure at hour {day*24 + hour_of_day}")
                    return None
            
            # Extract measurements
            buses = dss.Circuit.AllBusNames()
            vmag = dss.Circuit.AllBusMagPu()
            
            voltages = {}
            i = 0
            for bus in buses:
                dss.Circuit.SetActiveBus(bus)
                num_phases = dss.Bus.NumNodes()
                if num_phases > 0:
                    voltages[bus] = np.mean(vmag[i:i + num_phases])
                else:
                    voltages[bus] = 1.0  # Default voltage
                i += num_phases
            
            # Create record
            record = voltages.copy()
            record.update({
                "hour": hour_of_day,
                "day": day,
                "timestamp": f"day{day}_hour{hour_of_day:02d}",
                "pv_kw": total_pv_kw,
                "load_kw": total_load_kw,
                "fdia": 1 if fdia_info else 0,
                "fdia_type": fdia_info["type"] if fdia_info else 0,
                "fdia_target_bus": fdia_info["target"] if fdia_info else None,
                "fdia_severity": fdia_info.get("severity", 0) if fdia_info else 0
            })
            
            # Apply voltage-based FDIA attacks
            if fdia_info:
                record = apply_voltage_attacks(record, fdia_info, voltages)
            
            return record
    
    return None

def apply_voltage_attacks(record, fdia_info, original_voltages):
    """Apply voltage-based attacks"""
    attack_type = fdia_info["type"]
    target_bus = fdia_info["target"]
    severity = fdia_info.get("severity", 0.5)
    
    # Find target bus in voltage measurements
    target_key = None
    for key in original_voltages:
        if target_bus in key:
            target_key = key
            break
    
    if not target_key:
        return record
    
    if attack_type == 1:  # Voltage spike
        spike = 0.05 + severity * 0.15  # 5-20% increase
        record[target_key] = min(record[target_key] + spike, 1.5)
        
    elif attack_type == 2:  # Voltage sag
        drop = 0.05 + severity * 0.15  # 5-20% decrease
        record[target_key] = max(record[target_key] - drop, 0.8)
        
    elif attack_type == 4:  # Measurement noise
        noise_std = 0.01 + severity * 0.03  # 1-4% noise
        noise = np.random.normal(0, noise_std)
        record[target_key] += noise
        
    elif attack_type == 5:  # Coordinated attack (multiple buses)
        for key in original_voltages:
            if any(bus in key for bus in ["632", "633", "634"]):
                bias = np.random.uniform(-0.05, 0.05) * severity
                record[key] += bias
                
    elif attack_type == 6:  # Stealthy bias
        bias = np.random.uniform(-0.02, 0.02) * severity  # Very small bias
        record[target_key] += bias
    
    return record

dss.Basic.ClearAll()

try:
    dss.Command(r"Redirect data/raw/IEEE13Nodeckt.dss")
    print("IEEE 13-bus system loaded successfully")
except Exception as e:
    print(f"Failed to load IEEE 13-bus system: {e}")
    raise

num_buses = dss.Circuit.NumBuses()
print(f"System verified: {num_buses} buses")

if num_buses == 0:
    raise ValueError("No buses found - check DSS file")

# Create profiles
daily_solar, daily_load = create_realistic_profiles()

# Generate attack schedule
print("Generating FDIA attack schedule")
attack_gen = FDIAAttackGenerator()
fdia_schedule, total_attacks, type_counts = attack_gen.generate_attack_schedule(CONFIG['n_days'])

print(f"FDIA Schedule Summary:")
print(f"   Total attack hours: {total_attacks}/{total_hours} ({total_attacks/total_hours*100:.1f}%)")
for attack_type, count in type_counts.items():
    attack_name = attack_gen.attack_types[attack_type]["name"]
    print(f"   Type {attack_type} ({attack_name}): {count} hours")

# Run simulation
print(f"\nStarting simulation")
records = []
failed_hours = 0

for t in range(total_hours):
    hour_of_day = t % 24
    day = t // 24
    fdia_info = fdia_schedule.get(day, {}).get(hour_of_day)
    
    record = simulate_with_fdia(hour_of_day, day, fdia_info, daily_solar, daily_load)
    
    if record:
        records.append(record)
    else:
        failed_hours += 1
        # Create a basic record for failed simulations
        basic_record = {
            "hour": hour_of_day,
            "day": day,
            "timestamp": f"day{day}_hour{hour_of_day:02d}",
            "fdia": 1 if fdia_info else 0,
            "fdia_type": fdia_info["type"] if fdia_info else 0,
            "failed_simulation": True
        }
        records.append(basic_record)
    
    # Progress indicator
    if t % 240 == 0:  # Every 10 days
        progress = (t / total_hours) * 100
        print(f"Progress: {progress:.1f}% ({t}/{total_hours} hours, {failed_hours} failed)")

print(f"\nSimulation complete")
print(f"Success rate: {((total_hours - failed_hours) / total_hours * 100):.1f}%")

df = pd.DataFrame(records)

print(f"\nFinal Dataset Statistics:")
print(f"Total records: {len(df)}")
print(f"Failed simulations: {failed_hours}")
print(f"FDIA records: {df['fdia'].sum()}")
print(f"FDIA percentage: {(df['fdia'].sum() / len(df) * 100):.1f}%")

if 'fdia_type' in df.columns:
    fdia_counts = df.groupby('fdia_type').size()
    for attack_type in range(1, 7):
        if attack_type in fdia_counts:
            attack_name = attack_gen.attack_types[attack_type]["name"]
            print(f"Type {attack_type} ({attack_name}): {fdia_counts[attack_type]}")

filename = f"ieee13_fdia.csv"
df.to_csv(f"{CONFIG['output_dir']}/{filename}", index=False)

print(f"\nDataset saved: {filename}")
print(f"Saved to: {CONFIG['output_dir']}/")