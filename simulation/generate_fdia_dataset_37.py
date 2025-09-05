import opendssdirect as dss
import pandas as pd
import numpy as np
import random
import os

# === CONFIG ===
np.random.seed(42)
random.seed(42)

CONFIG = {
    'n_timesteps': 17520,  # 730 days * 24 hours (2 years)
    'fdia_probability': 0.1,
    'measurement_noise_std': 0.01,
    'attack_magnitude': 0.05,
    'convergence_tolerance': 1e-6,
    'max_retries': 3,
    'output_dir': 'data/processed',
    'n_buses': 37,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

print(f"Initializing FDIA dataset generation")
print(f"Timesteps: {CONFIG['n_timesteps']} ({CONFIG['n_timesteps']/24:.0f} days)")
print(f"FDIA probability: {CONFIG['fdia_probability']*100:.1f}%")

class StateEstimationFDIA:
    """
    Regular FDIA generation using state estimation framework
    Based on Liu et al. (2009) - False data injection attacks against state estimation
    """
    
    def __init__(self, n_buses=37):
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
        Compute realistic Jacobian matrix H for measurement function z = h(x)
        Creates better coupling between voltage magnitudes and angles
        """
        n_states = len(state_vector)
        n_measurements = n_states  # Assume full observability
        n_vmag = n_states // 2
        
        # Start with identity matrix
        H = np.eye(n_measurements, n_states)
        
        # Create realistic power system coupling
        for i in range(n_vmag):
            for j in range(n_vmag):
                if i != j:
                    # Voltage magnitude measurements depend on neighboring bus angles
                    if abs(i - j) <= 2:  # Neighboring buses
                        coupling_strength = 0.3 / (abs(i - j) + 1)
                        if j + n_vmag < n_states:
                            H[i, j + n_vmag] = coupling_strength
                    
                    # Angle measurements depend on neighboring magnitudes
                    if i + n_vmag < n_measurements and abs(i - j) <= 2:
                        coupling_strength = 0.2 / (abs(i - j) + 1)
                        H[i + n_vmag, j] = coupling_strength
        
        # Add stronger self-coupling between magnitude and angle of same bus
        for i in range(n_vmag):
            if i + n_vmag < n_states and i + n_vmag < n_measurements:
                H[i, i + n_vmag] = 0.6  # Magnitude depends on local angle
                H[i + n_vmag, i] = 0.4  # Angle depends on local magnitude
        
        # Add realistic measurement correlation
        correlation_factor = np.random.normal(1.0, 0.03, size=(n_measurements, n_states))
        H = H * correlation_factor
        
        # Ensure H is well-conditioned
        # Add small diagonal perturbation to avoid singularity
        H += np.eye(n_measurements, n_states) * 0.01
        
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
    
    def generate_fdia_attack_vector(self, H, attack_magnitude=None):
        if H is None:
            return None, None
            
        n_states = H.shape[1]
        n_measurements = H.shape[0]
        
        # Random attack magnitude (mode-dependent)
        if attack_magnitude is None:
            attack_magnitude = CONFIG['attack_magnitude']
        
        # Generate sophisticated attack vector c on state variables
        # Strategy 1: Target voltage magnitude states (more observable)
        n_vmag = n_states // 2
        
        # Create attack pattern with EXTREMELY high probability on voltage magnitudes
        attack_probs = np.ones(n_states) * 0.85  # Much higher base probability (was 0.7)
        attack_probs[:n_vmag] = 0.98  # Almost guaranteed attack on voltage magnitudes (was 0.95)
        
        # Generate attack mask
        attack_mask = np.random.random(n_states) < attack_probs
        n_attacked_states = np.sum(attack_mask)
        
        if n_attacked_states == 0:
            # Ensure at least one state is attacked
            attack_mask[np.random.randint(n_vmag)] = True
            n_attacked_states = 1
        
        c = np.zeros(n_states)
        
        # Generate SUBTLE, realistic attack patterns
        attacked_indices = np.where(attack_mask)[0]
        
        # Strategy: Subtle correlated attacks that are harder to detect
        base_attack = np.random.normal(0, attack_magnitude * 1.0, size=1)[0]  # Much more subtle (was 8.0)
        
        for i, idx in enumerate(attacked_indices):
            # Add correlation based on bus proximity
            correlation_factor = np.exp(-i * 0.1)  # Faster decay for more realistic correlation
            noise = np.random.normal(0, attack_magnitude * 0.3)  # Much less noise (was 1.5)
            c[idx] = base_attack * correlation_factor + noise
        
        # Strategy: Scale attacks based on measurement sensitivity (more realistic)
        measurement_sensitivity = np.abs(H).sum(axis=0)  # Sum of absolute values per state
        sensitivity_weights = measurement_sensitivity / np.max(measurement_sensitivity)
        
        # Scale attacks proportionally to sensitivity (more realistic than inverse)
        for idx in attacked_indices:
            if sensitivity_weights[idx] > 0:
                c[idx] *= (1.2 + 0.3 * sensitivity_weights[idx])  # Subtle scaling (was 4.0)
        
        # Compute attack vector a = Hc (ensures undetectability by residual test)
        attack_vector = H @ c
        
        # Strategy: Subtle amplification on select measurements
        critical_measurements = np.argsort(np.abs(attack_vector))[-n_measurements//4:]  # Only top 25%
        attack_vector[critical_measurements] *= 1.3  # Very modest amplification (was 4.0)
        
        attack_info = {
            'attack_magnitude': attack_magnitude,
            'attacked_states': attacked_indices.tolist(),
            'n_attacked_states': n_attacked_states,
            'attack_type': 'ultra_obvious_sophisticated',
            'base_attack': base_attack,
            'critical_measurements': critical_measurements.tolist()
        }
        
        return attack_vector, attack_info
    
    def apply_fdia_attack(self, measurements, attack_vector):
        """Apply FDIA attack: z' = z + a"""
        if measurements is None or attack_vector is None:
            return measurements
            
        return measurements + attack_vector

def simulate_timestep(timestep, fdia_generator):
    """Simulate single timestep with optional regular FDIA attack"""
    
    # Realistic load profile with daily and seasonal variations
    hour_of_day = timestep % 24
    day_of_year = (timestep // 24) % 365
    
    # Daily load pattern (peak during 7-9 AM and 5-8 PM)
    if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 20:
        daily_factor = 1.0  # Peak hours
    elif 22 <= hour_of_day or hour_of_day <= 5:
        daily_factor = 0.6  # Night hours
    else:
        daily_factor = 0.8  # Normal hours
    
    # Seasonal variation (higher in summer/winter due to heating/cooling)
    seasonal_factor = 0.85 + 0.15 * np.cos(2 * np.pi * day_of_year / 365)
    
    # Random daily variation (±5%)
    random_factor = 1.0 + np.random.normal(0, 0.05)
    
    # Combined load factor with realistic variations
    load_factor = 0.7 * daily_factor * seasonal_factor * random_factor
    load_factor = np.clip(load_factor, 0.4, 1.2)  # Reasonable bounds
    
    try:
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
        # Generate regular FDIA attack
        attack_vector, attack_info = fdia_generator.generate_fdia_attack_vector(H)
        
        z_attacked = fdia_generator.apply_fdia_attack(z_normal, attack_vector)
        
        # Calculate attack statistics
        attack_magnitude = np.linalg.norm(attack_vector)
        
    else:
        z_attacked = z_normal.copy()
        attack_vector = np.zeros_like(z_normal)
        attack_magnitude = 0.0
        attack_info = {'attack_type': 'none'}
    
    # Create record
    record = {
        'timestep': timestep,
        'hour_of_day': timestep % 24,
        'day': timestep // 24,
        'load_factor': load_factor,  # Constant load factor
        'fdia_label': 1 if has_attack else 0,
        'attack_magnitude': attack_magnitude,
        'attack_type': attack_info.get('attack_type', 'none'),
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
    dss.Basic.ClearAll()

    try:
        dss.Command(r"Redirect data/raw/IEEE37Nodeckt.dss")
        print("IEEE 37-bus system loaded successfully")
    except Exception as e:
        print(f"Failed to load IEEE 37-bus system: {e}")
        raise

    num_buses = dss.Circuit.NumBuses()
    print(f"System verified: {num_buses} buses")

    if num_buses == 0:
        raise ValueError("No buses found - check DSS file")

    fdia_generator = StateEstimationFDIA(n_buses=num_buses)

    print(f"\nStarting FDIA simulation")
    print(f"Generating {CONFIG['n_timesteps']} timesteps with {CONFIG['fdia_probability']*100:.1f}% FDIA probability")

    records = []
    attack_count = 0

    for t in range(CONFIG['n_timesteps']):
        record = simulate_timestep(t, fdia_generator)
        
        if record:
            records.append(record)
            if record['fdia_label'] == 1:
                attack_count += 1

    df = pd.DataFrame(records)

    print(f"\nFinal Dataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"FDIA records: {df['fdia_label'].sum()}")
    print(f"FDIA percentage: {(df['fdia_label'].sum() / len(df) * 100):.1f}%")

    filename = f"ieee37_fdia_dataset.csv"
    output_path = f"{CONFIG['output_dir']}/{filename}"
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved: {filename}")
    print(f"Saved to: {CONFIG['output_dir']}/")
    print(f"Dataset ready for LSTM-VAE-GAN training")