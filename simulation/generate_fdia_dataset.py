import opendssdirect as dss
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
from scipy.linalg import pinv

# === CONFIG ===
np.random.seed(42)
random.seed(42)

CONFIG = {
    'n_timesteps': 16800,  # 700 days * 24 hours
    'fdia_probability': 0.3,  # 30% of timesteps have FDIA attacks
    'measurement_noise_std': 0.01,  # 1% measurement noise
    'attack_magnitude_range': (0.02, 0.08),  # 2-8% attack magnitude
    'convergence_tolerance': 1e-6,
    'max_retries': 3,
    'output_dir': 'data/processed',
    'n_buses': 13,  # IEEE 13-bus system
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

print(f"Initializing Jacobian-based FDIA dataset generation")
print(f"Timesteps: {CONFIG['n_timesteps']} ({CONFIG['n_timesteps']/24:.0f} days)")
print(f"FDIA probability: {CONFIG['fdia_probability']*100:.1f}%")

class StateEstimationFDIA:
    """
    Realistic FDIA generation using state estimation framework
    Based on Liu et al. (2009) - False data injection attacks against state estimation
    """
    
    def __init__(self, n_buses=13):
        self.n_buses = n_buses
        self.measurement_matrix = None
        self.jacobian_matrix = None
        
    def extract_system_state(self):
        """Extract true system state (voltage magnitudes and angles)"""
        try:
            # Get voltage magnitudes (per unit)
            vmag = np.array(dss.Circuit.AllBusMagPu())
            
            # Get voltage angles - use bus iteration method
            vang = []
            bus_names = dss.Circuit.AllBusNames()
            
            for bus_name in bus_names:
                dss.Circuit.SetActiveBus(bus_name)
                # Get the voltage angle for the first phase of each bus
                # Use Bus.puVmagAngle which returns [mag1, ang1, mag2, ang2, ...]
                vm_ang = dss.Bus.puVmagAngle()
                if len(vm_ang) >= 2:
                    angle_deg = vm_ang[1]  # Second element is angle in degrees
                    vang.append(np.deg2rad(angle_deg))  # Convert to radians
                else:
                    vang.append(0.0)  # Default angle if no data
            
            vang = np.array(vang)
            
            # Combine into state vector [V_mag, V_ang]
            state_vector = np.concatenate([vmag, vang])
            
            return state_vector
            
        except Exception as e:
            print(f"Error extracting system state: {e}")
            return None
    
    def compute_measurement_jacobian(self, state_vector):
        """
        Compute Jacobian matrix H for measurement function z = h(x)
        Simplified approximation for demonstration
        """
        n_states = len(state_vector)
        n_measurements = n_states  # Assume full observability
        
        # Simplified Jacobian (identity-like with small perturbations)
        # In practice, this would be computed from power flow sensitivities
        H = np.eye(n_measurements, n_states)
        
        # Add some coupling between voltage magnitudes and angles
        n_vmag = n_states // 2
        for i in range(n_vmag):
            if i + n_vmag < n_states:
                H[i, i + n_vmag] = 0.1  # Small coupling
                H[i + n_vmag, i] = 0.05
        
        # Add measurement noise correlation
        noise_factor = np.random.normal(1.0, 0.02, size=(n_measurements, n_states))
        H = H * noise_factor
        
        return H
    
    def generate_measurements(self, state_vector, add_noise=True):
        """Generate measurements z = Hx + e"""
        if state_vector is None:
            return None, None
            
        # Compute measurement Jacobian
        H = self.compute_measurement_jacobian(state_vector)
        
        # Generate measurements
        z_true = H @ state_vector
        
        if add_noise:
            # Add Gaussian measurement noise
            noise = np.random.normal(0, CONFIG['measurement_noise_std'], size=z_true.shape)
            z_measured = z_true + noise
        else:
            z_measured = z_true.copy()
            
        return z_measured, H
    
    def generate_stealthy_attack_vector(self, H, attack_magnitude=None):
        """
        Generate stealthy FDIA attack vector a = Hc
        Attack is undetectable by residual-based bad data detection
        """
        if H is None:
            return None
            
        n_states = H.shape[1]
        
        # Random attack magnitude
        if attack_magnitude is None:
            attack_magnitude = np.random.uniform(*CONFIG['attack_magnitude_range'])
        
        # Generate random sparse attack vector c
        # Attack only a subset of states to maintain stealth
        n_attacked_states = max(1, int(0.3 * n_states))  # Attack 30% of states
        attacked_indices = np.random.choice(n_states, n_attacked_states, replace=False)
        
        c = np.zeros(n_states)
        c[attacked_indices] = np.random.normal(0, attack_magnitude, size=n_attacked_states)
        
        # Compute attack vector a = Hc (ensures stealth property)
        attack_vector = H @ c
        
        return attack_vector, c
    
    def apply_fdia_attack(self, measurements, attack_vector):
        """Apply FDIA attack: z' = z + a"""
        if measurements is None or attack_vector is None:
            return measurements
            
        return measurements + attack_vector

def create_load_profile(timestep):
    """Create realistic load profile based on time of day and season"""
    hour_of_day = timestep % 24
    day_of_year = (timestep // 24) % 365
    
    # Base load profile (daily pattern)
    if 6 <= hour_of_day <= 8 or 17 <= hour_of_day <= 22:  # Peak hours
        base_load = 0.85
    elif 22 <= hour_of_day or hour_of_day <= 6:  # Night hours
        base_load = 0.45
    else:  # Off-peak
        base_load = 0.65
    
    # Seasonal variation
    seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
    
    # Add random variation
    load_factor = base_load * seasonal_factor * (1 + np.random.normal(0, 0.05))
    
    return max(0.3, min(1.2, load_factor))

def simulate_timestep(timestep, fdia_generator):
    """Simulate single timestep with optional FDIA attack"""
    
    # Set realistic load based on time
    load_factor = create_load_profile(timestep)
    
    # Apply load to the system (simplified)
    try:
        # Set loads proportionally
        dss.Loads.First()
        while dss.Loads.Name():
            base_kw = dss.Loads.kW()
            dss.Loads.kW(base_kw * load_factor)
            if not dss.Loads.Next():
                break
        
        # Solve power flow
        dss.Solution.Solve()
        
        if not dss.Solution.Converged():
            return None
            
    except Exception as e:
        print(f"Power flow error at timestep {timestep}: {e}")
        return None
    
    # Extract system state
    state_vector = fdia_generator.extract_system_state()
    if state_vector is None:
        return None
    
    # Generate normal measurements
    z_normal, H = fdia_generator.generate_measurements(state_vector, add_noise=True)
    if z_normal is None:
        return None
    
    # Determine if this timestep has an attack
    has_attack = np.random.random() < CONFIG['fdia_probability']
    
    if has_attack:
        # Generate stealthy attack
        attack_vector, attack_state = fdia_generator.generate_stealthy_attack_vector(H)
        z_attacked = fdia_generator.apply_fdia_attack(z_normal, attack_vector)
        
        # Calculate attack statistics
        attack_magnitude = np.linalg.norm(attack_vector)
        attack_stealth = np.linalg.norm(attack_state) if attack_state is not None else 0
        
    else:
        z_attacked = z_normal.copy()
        attack_vector = np.zeros_like(z_normal)
        attack_magnitude = 0.0
        attack_stealth = 0.0
    
    # Create record
    record = {
        'timestep': timestep,
        'hour_of_day': timestep % 24,
        'day': timestep // 24,
        'load_factor': load_factor,
        'fdia_label': 1 if has_attack else 0,
        'attack_magnitude': attack_magnitude,
        'attack_stealth': attack_stealth,
        'measurement_noise_std': CONFIG['measurement_noise_std']
    }
    
    # Add measurements (normal and attacked)
    for i, (z_norm, z_att) in enumerate(zip(z_normal, z_attacked)):
        record[f'z_normal_{i}'] = z_norm
        record[f'z_attacked_{i}'] = z_att
        record[f'attack_vector_{i}'] = attack_vector[i]
    
    return record

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Initialize OpenDSS and load IEEE 13-bus system
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

    # Initialize FDIA generator
    fdia_generator = StateEstimationFDIA(n_buses=num_buses)

    # Run simulation
    print(f"\nStarting FDIA simulation")
    print(f"Generating {CONFIG['n_timesteps']} timesteps with {CONFIG['fdia_probability']*100:.1f}% FDIA probability")

    records = []
    failed_timesteps = 0
    attack_count = 0

    for t in range(CONFIG['n_timesteps']):
        record = simulate_timestep(t, fdia_generator)
        
        if record:
            records.append(record)
            if record['fdia_label'] == 1:
                attack_count += 1
        else:
            failed_timesteps += 1
            # Create basic record for failed simulations
            basic_record = {
                'timestep': t,
                'hour_of_day': t % 24,
                'day': t // 24,
                'fdia_label': 0,
                'failed_simulation': True
            }
            records.append(basic_record)
        
        # Progress indicator
        if t % 1000 == 0 and t > 0:
            progress = (t / CONFIG['n_timesteps']) * 100
            current_attack_rate = (attack_count / t) * 100
            print(f"Progress: {progress:.1f}% ({t}/{CONFIG['n_timesteps']} timesteps)")
            print(f"   Success: {((t - failed_timesteps) / t * 100):.1f}%, FDIA: {current_attack_rate:.1f}%")

    print(f"\nSimulation complete")
    print(f"Total timesteps: {CONFIG['n_timesteps']}")
    print(f"Failed simulations: {failed_timesteps}")
    print(f"Success rate: {((CONFIG['n_timesteps'] - failed_timesteps) / CONFIG['n_timesteps'] * 100):.1f}%")

    # Create DataFrame
    df = pd.DataFrame(records)

    print(f"\nFinal Dataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"FDIA records: {df['fdia_label'].sum()}")
    print(f"FDIA percentage: {(df['fdia_label'].sum() / len(df) * 100):.1f}%")

    # Calculate statistics for attacked measurements
    if 'attack_magnitude' in df.columns:
        attacked_df = df[df['fdia_label'] == 1]
        if len(attacked_df) > 0:
            print(f"Average attack magnitude: {attacked_df['attack_magnitude'].mean():.6f}")
            print(f"Attack magnitude range: {attacked_df['attack_magnitude'].min():.6f} - {attacked_df['attack_magnitude'].max():.6f}")

    # Save dataset
    filename = f"ieee13_fdia_dataset.csv"
    output_path = f"{CONFIG['output_dir']}/{filename}"
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved: {filename}")
    print(f"Saved to: {CONFIG['output_dir']}/")

