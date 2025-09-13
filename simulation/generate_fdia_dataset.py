import opendssdirect as dss
import pandas as pd
import numpy as np
import random
import os

np.random.seed(42)
random.seed(42)

CONFIG = {
    'n_timesteps': 17520,  # 730 days * 24 hours (2 years)
    'fdia_probability': 0.1,
    'measurement_noise_std': 0.1,
    'attack_magnitude': 0.05,
    'convergence_tolerance': 1e-6, 
    'max_retries': 3,
    'output_dir': 'data/processed',
    'n_buses': 13,
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
    
    def __init__(self, n_buses=13):
        self.n_buses = n_buses
        self.measurement_matrix = None
        self.jacobian_matrix = None
        self.attack_state = {
            'active_campaign': None,
            'campaign_progress': 0,
            'target_states': [],
            'buildup_factor': 1.0,
            'last_attack_time': -100
        }
        
    def extract_system_state(self):
        try:
            vmag = np.array(dss.Circuit.AllBusMagPu())
            
            vang = []
            bus_names = dss.Circuit.AllBusNames()
            
            for bus_name in bus_names:
                dss.Circuit.SetActiveBus(bus_name)
                # Get the voltage angle for the first phase of each bus
                # Use Bus.puVmagAngle which returns [mag1, ang1, mag2, ang2, ...]
                vm_ang = dss.Bus.puVmagAngle()
                if len(vm_ang) >= 2:
                    angle_deg = vm_ang[1] 
                    vang.append(np.deg2rad(angle_deg))
                else:
                    vang.append(0.0)  # Default angle if no data
            
            vang = np.array(vang)
            
            state_vector = np.concatenate([vmag, vang])
            
            return state_vector
            
        except Exception as e:
            print(f"Error extracting system state: {e}")
            return None
    
    def compute_measurement_jacobian(self, state_vector):
        n_states = len(state_vector)
        n_measurements = n_states
        n_vmag = n_states // 2
        
        H = np.eye(n_measurements, n_states)
        
        for i in range(n_vmag):
            for j in range(n_vmag):
                if i != j:
                    # Voltage magnitude measurements depend on neighboring bus angles
                    if abs(i - j) <= 2: 
                        coupling_strength = 0.3 / (abs(i - j) + 1)
                        if j + n_vmag < n_states:
                            H[i, j + n_vmag] = coupling_strength
                    
                    # Angle measurements depend on neighboring magnitudes
                    if i + n_vmag < n_measurements and abs(i - j) <= 2:
                        coupling_strength = 0.2 / (abs(i - j) + 1)
                        H[i + n_vmag, j] = coupling_strength
        
        for i in range(n_vmag):
            if i + n_vmag < n_states and i + n_vmag < n_measurements:
                H[i, i + n_vmag] = 0.6
                H[i + n_vmag, i] = 0.4 
        
        # Add realistic measurement correlation
        correlation_factor = np.random.normal(1.0, 0.005, size=(n_measurements, n_states))
        H = H * correlation_factor
        
        # Add small diagonal perturbation to avoid singularity
        H += np.eye(n_measurements, n_states) * 0.01
        
        return H
    
    def generate_measurements(self, state_vector, add_noise=True):
        if state_vector is None:
            return None, None
            
        H = self.compute_measurement_jacobian(state_vector)
        
        z_true = H @ state_vector
        
        if add_noise:
            noise = np.random.normal(0, CONFIG['measurement_noise_std'], size=z_true.shape)
            z_measured = z_true + noise
        else:
            z_measured = z_true.copy()
            
        return z_measured, H
    
    def generate_fdia_attack_vector(self, H, timestep=0, load_factor=1.0, measurement_history=None):
        if H is None:
            return None, None
            
        n_states = H.shape[1]
        n_measurements = H.shape[0]
        n_vmag = n_states // 2
        
        attack_magnitude = CONFIG['attack_magnitude']
        hour_of_day = timestep % 24
        
        # === STRATEGY 0: MULTI-STAGE CAMPAIGN ATTACKS ===
        campaign_attack = False
        time_since_last_attack = timestep - self.attack_state['last_attack_time']
        
        if (self.attack_state['active_campaign'] is None and np.random.random() < 0.02 and time_since_last_attack > 48): 
            self.attack_state['active_campaign'] = 'gradual_buildup'
            self.attack_state['campaign_progress'] = 0
            self.attack_state['target_states'] = np.random.choice(range(n_vmag), size=min(3, n_vmag//2), replace=False).tolist()
            self.attack_state['buildup_factor'] = 0.1
            campaign_attack = True
            
        elif self.attack_state['active_campaign'] is not None:
            self.attack_state['campaign_progress'] += 1
            self.attack_state['buildup_factor'] = min(2.0, 0.1 + (self.attack_state['campaign_progress'] * 0.05))
            
            if np.random.random() < 0.4:
                campaign_attack = True
                
            if (self.attack_state['campaign_progress'] > 20 or np.random.random() < 0.1):
                self.attack_state['active_campaign'] = None
                
        if campaign_attack:
            self.attack_state['last_attack_time'] = timestep
        
        # === STRATEGY 1: TIME-AWARE STEALTH ===
        stealth_multiplier = 1.0
        if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 20:  # Peak hours
            stealth_multiplier = 2.0
        elif 22 <= hour_of_day or hour_of_day <= 5:  # Low activity hours
            stealth_multiplier = 0.4
        
        if campaign_attack:
            stealth_multiplier *= self.attack_state['buildup_factor']
        
        # === STRATEGY 2: MEASUREMENT-NOISE CAMOUFLAGE ===
        if measurement_history is not None and len(measurement_history) > 5:
            recent_measurements = np.array(measurement_history[-5:])
            measurement_variance = np.var(recent_measurements, axis=0)
            noise_adaptive_scaling = np.sqrt(measurement_variance) / CONFIG['measurement_noise_std']
            noise_adaptive_scaling = np.clip(noise_adaptive_scaling, 0.3, 3.0)
        else:
            noise_adaptive_scaling = np.ones(n_measurements)
        
        # === STRATEGY 3: SMART STATE TARGETING ===
        if campaign_attack and self.attack_state['target_states']:
            attacked_indices = np.array(self.attack_state['target_states'])
            attack_mask = np.zeros(n_states, dtype=bool)
            attack_mask[attacked_indices] = True
        else:
            state_vulnerability = np.abs(H).std(axis=0)
            vulnerability_weights = state_vulnerability / np.max(state_vulnerability)
            
            observability_penalty = 1.0 / (vulnerability_weights + 0.1)
            target_probs = observability_penalty / np.sum(observability_penalty)
            
            attack_mask = np.random.random(n_states) < (target_probs * 0.6)
            attacked_indices = np.where(attack_mask)[0]
        
        n_attacked_states = len(attacked_indices)
        
        if n_attacked_states == 0:
            state_vulnerability = np.abs(H).std(axis=0)
            vulnerability_weights = state_vulnerability / np.max(state_vulnerability)
            least_observable_idx = np.argmin(vulnerability_weights[:n_vmag])
            attack_mask[least_observable_idx] = True
            attacked_indices = np.array([least_observable_idx])
            n_attacked_states = 1
        
        c = np.zeros(n_states)
        
        # === STRATEGY 4: CORRELATED PERTURBATIONS MIMICKING NATURAL VARIATIONS ===
        if len(attacked_indices) > 1:
            correlation_matrix = np.corrcoef(H.T)
            correlation_matrix = np.nan_to_num(correlation_matrix, 0)
            
            strategic_idx = attacked_indices[0] 
            base_attack = np.random.normal(0, attack_magnitude * stealth_multiplier * 0.7)
            c[strategic_idx] = base_attack
            
            for idx in attacked_indices[1:]:
                correlation_factor = correlation_matrix[strategic_idx, idx]
                correlated_noise = np.random.normal(0, attack_magnitude * 0.2)
                c[idx] = base_attack * abs(correlation_factor) * 0.6 + correlated_noise
        else:
            idx = attacked_indices[0]
            c[idx] = np.random.normal(0, attack_magnitude * stealth_multiplier * 0.5)
        
        # === STRATEGY 5: MEASUREMENT-SPACE CAMOUFLAGE ===
        attack_vector = H @ c
        
        for i in range(len(attack_vector)):
            noise_scale = noise_adaptive_scaling[i] if i < len(noise_adaptive_scaling) else 1.0
            attack_vector[i] *= noise_scale
        
        attack_info = {
            'attack_magnitude': np.linalg.norm(c),
            'attacked_states': attacked_indices.tolist(),
            'n_attacked_states': n_attacked_states,
            'attack_type': 'campaign_stealth' if campaign_attack else 'time_aware_stealth',
            'stealth_multiplier': stealth_multiplier,
            'hour_of_day': hour_of_day,
            'strategic_targeting': True,
            'correlation_based': len(attacked_indices) > 1,
            'campaign_active': campaign_attack,
            'campaign_progress': self.attack_state['campaign_progress'] if campaign_attack else 0
        }
        
        return attack_vector, attack_info
    
    def apply_fdia_attack(self, measurements, attack_vector):
        """Apply FDIA attack: z' = z + a"""
        if measurements is None or attack_vector is None:
            return measurements
            
        return measurements + attack_vector

def simulate_timestep(timestep, fdia_generator, measurement_history=None):    
    if measurement_history is None:
        measurement_history = []
    
    hour_of_day = timestep % 24
    
    # Morning peak (7-9 AM) and evening peak (5-8 PM)
    morning_peak = 0.3 * np.exp(-((hour_of_day - 8)**2) / (2 * 1.5**2))  # Gaussian around 8 AM
    evening_peak = 0.4 * np.exp(-((hour_of_day - 18.5)**2) / (2 * 2**2))  # Gaussian around 6:30 PM
    
    # Base daily cycle (sinusoidal with minimum at 4 AM, maximum at 2 PM)
    base_cycle = 0.7 + 0.2 * np.sin(2 * np.pi * (hour_of_day - 4) / 24)
    
    daily_factor = (base_cycle + morning_peak + evening_peak)
    
    load_factor = 0.75 * daily_factor
    
    dss.Loads.First()
    while dss.Loads.Name():
        base_kw = dss.Loads.kW()
        dss.Loads.kW(base_kw * load_factor)
        if not dss.Loads.Next():
            break
    
    dss.Solution.Solve()
    
    state_vector = fdia_generator.extract_system_state()
    
    z_normal, H = fdia_generator.generate_measurements(state_vector, add_noise=True)
    
    measurement_history.append(z_normal.copy())
    if len(measurement_history) > 20:
        measurement_history.pop(0)
    
    base_attack_prob = CONFIG['fdia_probability']
    
    if len(measurement_history) >= 3:
        recent_variance = np.var([measurement_history[-1], measurement_history[-2], measurement_history[-3]], axis=0)
        high_variance_bonus = np.mean(recent_variance) * 100 
        adaptive_attack_prob = base_attack_prob * (1 + high_variance_bonus)
    else:
        adaptive_attack_prob = base_attack_prob
    
    if load_factor < 0.6 or (2 <= hour_of_day <= 6):  # Low activity periods
        adaptive_attack_prob *= 0.3
        
    adaptive_attack_prob = np.clip(adaptive_attack_prob, 0.005, 0.15)
    
    has_attack = np.random.random() < adaptive_attack_prob
    
    if has_attack:
        attack_vector, attack_info = fdia_generator.generate_fdia_attack_vector(H, timestep=timestep, load_factor=load_factor, measurement_history=measurement_history)
        
        z_attacked = fdia_generator.apply_fdia_attack(z_normal, attack_vector)
        
        attack_magnitude = np.linalg.norm(attack_vector)
        
    else:
        z_attacked = z_normal.copy()
        attack_vector = np.zeros_like(z_normal)
        attack_magnitude = 0.0
        attack_info = {'attack_type': 'none'}
    
    record = {
        'timestep': timestep,
        'hour_of_day': timestep % 24,
        'day': timestep // 24,
        'load_factor': load_factor,
        'fdia_label': 1 if has_attack else 0,
        'attack_magnitude': attack_magnitude,
        'attack_type': attack_info.get('attack_type', 'none'),
        'measurement_noise_std': CONFIG['measurement_noise_std']
    }
    
    for i, (z_norm, z_att) in enumerate(zip(z_normal, z_attacked)):
        record[f'z_normal_{i}'] = z_norm
        record[f'z_attacked_{i}'] = z_att
        record[f'attack_vector_{i}'] = attack_vector[i]
    
    return record

if __name__ == "__main__":
    dss.Basic.ClearAll()

    try:
        dss.Command(r"Redirect data/raw/IEEE13Nodeckt.dss")
    except Exception as e:
        print(f"Failed to load IEEE 13-bus system: {e}")
        raise

    num_buses = dss.Circuit.NumBuses()
    fdia_generator = StateEstimationFDIA(n_buses=num_buses)

    records = []
    attack_count = 0
    measurement_history = []

    for t in range(CONFIG['n_timesteps']):
        record = simulate_timestep(t, fdia_generator, measurement_history)
        
        if record:
            records.append(record)
            if record['fdia_label'] == 1:
                attack_count += 1

    df = pd.DataFrame(records)

    print(f"\nFinal Dataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"FDIA records: {df['fdia_label'].sum()}")
    print(f"FDIA percentage: {(df['fdia_label'].sum() / len(df) * 100):.1f}%")

    filename = f"ieee13_fdia_dataset.csv"
    output_path = f"{CONFIG['output_dir']}/{filename}"
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved: {filename}")
    print(f"Saved to: {CONFIG['output_dir']}/")