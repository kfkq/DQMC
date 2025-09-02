#!/usr/bin/env python3
"""
Jackknife and error estimation analysis for binned data from DQMC simulations.
For unequal-time observables, statr0.dat contains data at rx=0, ry=0 (not averaged over all r).
"""

import h5py
import numpy as np
import os
import glob
import argparse
import configparser
from pathlib import Path


def is_pt_enabled(param_file="parameters.in"):
    """
    Check the parameters.in file to see if Parallel Tempering is enabled.
    """
    if not os.path.exists(param_file):
        print(f"Warning: '{param_file}' not found. Assuming standard run (not Parallel Tempering).")
        return False
    
    config = configparser.ConfigParser()
    try:
        config.read(param_file)
        if 'ParallelTempering' in config and 'enabled' in config['ParallelTempering']:
            return config['ParallelTempering'].getboolean('enabled')
    except Exception as e:
        print(f"Warning: Could not parse '{param_file}'. Assuming standard run. Error: {e}")
    
    return False


def load_scalar_data(results_dir="results", pt_enabled=False):
    """
    Load scalar observable data from MPI rank files.
    
    Args:
        results_dir (str): Path to the results directory
        pt_enabled (bool): If True, only load data from data_0.h5
        
    Returns:
        dict: Dictionary with observable names as keys and arrays of binned data as values
    """
    if pt_enabled:
        print("Parallel Tempering run detected. Analyzing data_0.h5 only.")
        data_files = [os.path.join(results_dir, "data_0.h5")]
    else:
        print("Standard run detected. Aggregating data from all data_*.h5 files.")
        data_files = glob.glob(os.path.join(results_dir, "data_*.h5"))
    
    if not os.path.exists(data_files[0]):
        raise FileNotFoundError(f"No data files found in {results_dir}")
    
    # Dictionary to store data for each observable
    scalar_data = {}
    
    # Process each file
    for file_path in data_files:
        with h5py.File(file_path, 'r') as f:
            # Get all bin groups (format: bin_N)
            bin_groups = [key for key in f.keys() if key.startswith('bin_') and not key.startswith('binK_')]
            bin_groups.sort(key=lambda x: int(x.split('_')[1]))  # Sort by bin number
            
            # Process each bin
            for bin_group in bin_groups:
                group = f[bin_group]
                
                # Check if scalar group exists
                if 'scalar' in group:
                    scalar_group = group['scalar']
                    
                    # Process each scalar observable in the bin
                    for obs_name in scalar_group.keys():
                        # Check if this is a scalar dataset (0-dimensional or 1-element)
                        dataset = scalar_group[obs_name]
                        if len(dataset.shape) == 0 or (len(dataset.shape) == 1 and dataset.shape[0] == 1):
                            # Get scalar value
                            if len(dataset.shape) == 0:
                                value = dataset[()]  # 0-dimensional
                            else:
                                value = dataset[0]  # 1-element array
                            
                            # Initialize list for this observable if not already done
                            if obs_name not in scalar_data:
                                scalar_data[obs_name] = []
                            
                            scalar_data[obs_name].append(value)
    
    # Convert lists to numpy arrays
    for obs_name in scalar_data:
        scalar_data[obs_name] = np.array(scalar_data[obs_name])
    
    return scalar_data


def load_equaltime_data(results_dir="results", pt_enabled=False):
    """
    Load equal-time observable data from MPI rank files.
    
    Args:
        results_dir (str): Path to the results directory
        pt_enabled (bool): If True, only load data from data_0.h5
        
    Returns:
        tuple: (real_space_data, k_space_data) dictionaries
    """
    if pt_enabled:
        data_files = [os.path.join(results_dir, "data_0.h5")]
    else:
        data_files = glob.glob(os.path.join(results_dir, "data_*.h5"))
    
    if not os.path.exists(data_files[0]):
        raise FileNotFoundError(f"No data files found in {results_dir}")
    
    # Dictionary to store data for each observable
    real_space_data = {}
    k_space_data = {}
    
    # Process each file
    for file_path in data_files:
        with h5py.File(file_path, 'r') as f:
            # Get all bin groups (format: bin_N)
            bin_groups = [key for key in f.keys() if key.startswith('bin_') and not key.startswith('binK_')]
            bin_groups.sort(key=lambda x: int(x.split('_')[1]))  # Sort by bin number
            
            # Get all k-space bin groups (format: binK_N)
            k_bin_groups = [key for key in f.keys() if key.startswith('binK_')]
            k_bin_groups.sort(key=lambda x: int(x.split('_')[1]))  # Sort by bin number
            
            # Process real space bins
            for bin_group in bin_groups:
                group = f[bin_group]
                
                # Check if equaltime group exists
                if 'equaltime' in group:
                    eqtime_group = group['equaltime']
                    
                    # Process each equal-time observable in the bin
                    for obs_name in eqtime_group.keys():
                        value = np.array(eqtime_group[obs_name])
                        if obs_name not in real_space_data:
                            real_space_data[obs_name] = []
                        real_space_data[obs_name].append(value)
            
            # Process k-space bins
            for bin_group in k_bin_groups:
                group = f[bin_group]
                
                # Check if equaltime group exists
                if 'equaltime' in group:
                    eqtime_group = group['equaltime']
                    
                    # Process each equal-time observable in the bin
                    for obs_name in eqtime_group.keys():
                        value = np.array(eqtime_group[obs_name])
                        if obs_name not in k_space_data:
                            k_space_data[obs_name] = []
                        k_space_data[obs_name].append(value)
    
    return real_space_data, k_space_data


def load_unequaltime_data(results_dir="results", pt_enabled=False):
    """
    Load unequal-time observable data from MPI rank files.
    
    Args:
        results_dir (str): Path to the results directory
        pt_enabled (bool): If True, only load data from data_0.h5
        
    Returns:
        tuple: (real_space_data, k_space_data) dictionaries
    """
    if pt_enabled:
        data_files = [os.path.join(results_dir, "data_0.h5")]
    else:
        data_files = glob.glob(os.path.join(results_dir, "data_*.h5"))
    
    if not os.path.exists(data_files[0]):
        raise FileNotFoundError(f"No data files found in {results_dir}")
    
    # Dictionary to store data for each observable
    real_space_data = {}
    k_space_data = {}
    
    # Process each file
    for file_path in data_files:
        with h5py.File(file_path, 'r') as f:
            # Get all bin groups (format: bin_N)
            bin_groups = [key for key in f.keys() if key.startswith('bin_') and not key.startswith('binK_')]
            bin_groups.sort(key=lambda x: int(x.split('_')[1]))  # Sort by bin number
            
            # Get all k-space bin groups (format: binK_N)
            k_bin_groups = [key for key in f.keys() if key.startswith('binK_')]
            k_bin_groups.sort(key=lambda x: int(x.split('_')[1]))  # Sort by bin number
            
            # Process real space bins
            for bin_group in bin_groups:
                group = f[bin_group]
                
                # Check if unequaltime group exists
                if 'unequaltime' in group:
                    uneqtime_group = group['unequaltime']
                    
                    # Process each unequal-time observable in the bin
                    for obs_name in uneqtime_group.keys():
                        value = np.array(uneqtime_group[obs_name])
                        if obs_name not in real_space_data:
                            real_space_data[obs_name] = []
                        real_space_data[obs_name].append(value)
            
            # Process k-space bins
            for bin_group in k_bin_groups:
                group = f[bin_group]
                
                # Check if unequaltime group exists
                if 'unequaltime' in group:
                    uneqtime_group = group['unequaltime']
                    
                    # Process each unequal-time observable in the bin
                    for obs_name in uneqtime_group.keys():
                        value = np.array(uneqtime_group[obs_name])
                        if obs_name not in k_space_data:
                            k_space_data[obs_name] = []
                        k_space_data[obs_name].append(value)
    
    return real_space_data, k_space_data

# ... (jackknife and other helper functions remain unchanged) ...
def jackknife_analysis(data):
    """
    Perform jackknife analysis on data.
    
    Args:
        data (np.array): Array of binned measurements
        
    Returns:
        tuple: (mean, error) from jackknife analysis
    """
    n_bins = len(data)
    if n_bins < 2:
        raise ValueError("Need at least 2 bins for jackknife analysis")
    
    # Calculate full sample mean
    full_mean = np.mean(data)
    
    # Calculate jackknife estimators
    jackknife_means = np.zeros(n_bins)
    for i in range(n_bins):
        # Mean with ith bin removed
        jackknife_means[i] = (n_bins * full_mean - data[i]) / (n_bins - 1)
    
    # Calculate jackknife error
    # Error = sqrt( (n-1)/n * sum( (theta_i - theta_jk)^2 ) )
    # where theta_jk is the jackknife mean
    jackknife_mean = np.mean(jackknife_means)
    variance = np.sum((jackknife_means - jackknife_mean)**2) * (n_bins - 1) / n_bins
    error = np.sqrt(variance)
    
    return full_mean, error


def jackknife_analysis_array(data):
    """
    Perform jackknife analysis on array data (for equal-time and unequal-time observables).
    
    Args:
        data (list): List of arrays (binned measurements)
        
    Returns:
        tuple: (mean_array, error_array) from jackknife analysis
    """
    n_bins = len(data)
    if n_bins < 2:
        raise ValueError("Need at least 2 bins for jackknife analysis")
    
    # Convert to numpy array for easier manipulation
    data_array = np.array(data)
    
    # Calculate full sample mean
    full_mean = np.mean(data_array, axis=0)
    
    # Calculate jackknife estimators
    jackknife_means = np.zeros((n_bins,) + data_array.shape[1:], dtype=data_array.dtype)
    for i in range(n_bins):
        # Mean with ith bin removed
        jackknife_means[i] = (n_bins * full_mean - data_array[i]) / (n_bins - 1)
    
    # Calculate jackknife error
    # Error = sqrt( (n-1)/n * sum( (theta_i - theta_jk)^2 ) )
    # where theta_jk is the jackknife mean
    jackknife_mean = np.mean(jackknife_means, axis=0)
    variance = np.sum((jackknife_means - jackknife_mean[np.newaxis, ...])**2, axis=0) * (n_bins - 1) / n_bins
    error = np.sqrt(variance)
    
    return full_mean, error


def load_lattice_info(results_dir="results"):
    """
    Load lattice information from the info file.
    
    Args:
        results_dir (str): Path to the results directory
        
    Returns:
        dict: Dictionary with lattice parameters
    """
    info_file = os.path.join(results_dir, "info")
    lattice_info = {}
    
    try:
        with open(info_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    # Try to convert to int or float
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string
                    lattice_info[key] = value
    except FileNotFoundError:
        print(f"Error: Lattice info file {info_file} not found.")
        print("This file is required for coordinate conversion.")
        print("Please run the simulation first to generate this file.")
        exit(1)
    
    return lattice_info


def convert_r_indices_to_physical(x_idx, y_idx, lattice_info):
    """
    Convert real-space indices to physical positions.
    
    Args:
        x_idx (int): x index
        y_idx (int): y index
        lattice_info (dict): Lattice parameters
        
    Returns:
        tuple: (rx, ry) physical positions
    """
    L1 = lattice_info['L1']
    L2 = lattice_info['L2']
    a1_x = lattice_info['a1_x']
    a1_y = lattice_info['a1_y']
    a2_x = lattice_info['a2_x']
    a2_y = lattice_info['a2_y']
    
    # Convert indices to physical positions
    dx_phys = (x_idx - (L1/2 - 1)) * a1_x + (y_idx - (L2/2 - 1)) * a2_x
    dy_phys = (x_idx - (L1/2 - 1)) * a1_y + (y_idx - (L2/2 - 1)) * a2_y
    
    return dx_phys, dy_phys


def convert_k_indices_to_physical(kx_idx, ky_idx, lattice_info):
    """
    Convert k-space indices to physical k-vectors.
    
    Args:
        kx_idx (int): kx index
        ky_idx (int): ky index
        lattice_info (dict): Lattice parameters
        
    Returns:
        tuple: (kx, ky) physical k-vectors
    """
    L1 = lattice_info['L1']
    L2 = lattice_info['L2']
    
    # Compute reciprocal lattice vectors
    a1_x = lattice_info['a1_x']
    a1_y = lattice_info['a1_y']
    a2_x = lattice_info['a2_x']
    a2_y = lattice_info['a2_y']
    
    # Compute determinant
    det = a1_x * a2_y - a1_y * a2_x
    
    # Compute reciprocal lattice vectors
    b1_x = 2 * np.pi * a2_y / det / L1
    b1_y = -2 * np.pi * a2_x / det / L1
    b2_x = -2 * np.pi * a1_y / det / L2
    b2_y = 2 * np.pi * a1_x / det / L2
    
    # Convert indices to physical k-vectors
    # k-points are shifted to (-π, π] : range −L/2+1 … L/2
    qx = kx_idx - (L1 // 2) + 1
    qy = ky_idx - (L2 // 2) + 1
    
    kx_phys = qx * b1_x + qy * b2_x
    ky_phys = qx * b1_y + qy * b2_y
    
    return kx_phys, ky_phys


def analyze_observables(scalar_data, eqtime_data_r=None, eqtime_data_k=None, 
                       uneqtime_data_r=None, uneqtime_data_k=None, lattice_info=None):
    """
    Perform jackknife analysis on all observables.
    
    Args:
        scalar_data (dict): Dictionary with scalar observable data
        eqtime_data_r (dict): Dictionary with real-space equal-time observable data
        eqtime_data_k (dict): Dictionary with k-space equal-time observable data
        uneqtime_data_r (dict): Dictionary with real-space unequal-time observable data
        uneqtime_data_k (dict): Dictionary with k-space unequal-time observable data
        lattice_info (dict): Lattice parameters for coordinate conversion
    """
    results = {}
    
    # Analyze scalar observables
    for obs_name, data in scalar_data.items():
        try:
            mean, error = jackknife_analysis(data)
            results[obs_name] = (mean, error)
        except Exception as e:
            print(f"Error analyzing scalar observable {obs_name}: {e}")
            results[obs_name] = (np.nan, np.nan)
    
    # Write scalar results to file (default filename)
    output_filename = "scalarObservables.dat"
    if results:
        with open(output_filename, 'w') as f:
            # Write header
            f.write("# Observable Mean Error\n")
            # Write data in a parseable format
            for obs_name, (mean, error) in results.items():
                f.write(f"{obs_name} {mean} {error}\n")
    
    # Analyze equal-time observables (both real space and k-space)
    if eqtime_data_r and eqtime_data_k and lattice_info:
        # Process real space equal-time observables
        for obs_name, data in eqtime_data_r.items():
            try:
                mean, error = jackknife_analysis_array(data)
                
                # Create directory for this observable if it doesn't exist
                obs_dir = obs_name
                if not os.path.exists(obs_dir):
                    os.makedirs(obs_dir)
                
                # Write real space data to statr.dat
                statr_file = os.path.join(obs_dir, "statr.dat")
                with open(statr_file, 'w') as f:
                    f.write(f"# Equal-time observable: {obs_name} (Real space)\n")
                    f.write(f"# Dimensions: {mean.shape}\n")
                    f.write("# Format: rx ry a b mean error\n")
                    
                    # Assuming mean and error have shape (nx, ny, norb) where norb = ntau
                    nx, ny, norb = mean.shape
                    
                    for x in range(nx):
                        for y in range(ny):
                            for orb_idx in range(norb):
                                # Convert orbital index to a, b indices
                                a = orb_idx // lattice_info['n_orb'] if 'n_orb' in lattice_info else 0
                                b = orb_idx % lattice_info['n_orb'] if 'n_orb' in lattice_info else orb_idx
                                
                                # Convert indices to physical positions
                                rx, ry = convert_r_indices_to_physical(x, y, lattice_info)
                                f.write(f"{rx:12.6f} {ry:12.6f} {a:3d} {b:3d} {mean[x, y, orb_idx]:15.8e} {error[x, y, orb_idx]:15.8e}\n")
                
            except Exception as e:
                print(f"Error analyzing real-space equal-time observable {obs_name}: {e}")
        
        # Process k-space equal-time observables
        for obs_name, data in eqtime_data_k.items():
            try:
                # For complex data, we need to handle real and imaginary parts
                # Convert to complex array first
                # Data is stored as [nx, ny, ntau, 2] where last dimension is [real, imag]
                complex_data = []
                for d in data:
                    real_part = d[:, :, :, 0]
                    imag_part = d[:, :, :, 1]
                    complex_d = real_part + 1j * imag_part
                    complex_data.append(complex_d)
                
                mean, error = jackknife_analysis_array(complex_data)
                
                # Create directory for this observable if it doesn't exist
                obs_dir = obs_name
                if not os.path.exists(obs_dir):
                    os.makedirs(obs_dir)
                
                # Write k-space data to statk.dat
                statk_file = os.path.join(obs_dir, "statk.dat")
                with open(statk_file, 'w') as f:
                    f.write(f"# Equal-time observable: {obs_name} (K-space)\n")
                    f.write(f"# Dimensions: {mean.shape}\n")
                    f.write("# Format: kx ky a b mean_real mean_imag error_real error_imag\n")
                    
                    # Assuming mean and error have shape (nkx, nky, norb) where norb = ntau
                    nkx, nky, norb = mean.shape
                    
                    for kx in range(nkx):
                        for ky in range(nky):
                            for orb_idx in range(norb):
                                # Convert orbital index to a, b indices
                                a = orb_idx // lattice_info['n_orb'] if 'n_orb' in lattice_info else 0
                                b = orb_idx % lattice_info['n_orb'] if 'n_orb' in lattice_info else orb_idx
                                
                                # Convert indices to physical k-vectors
                                kx_phys, ky_phys = convert_k_indices_to_physical(kx, ky, lattice_info)
                                mean_val = mean[kx, ky, orb_idx]
                                error_val = error[kx, ky, orb_idx]
                                f.write(f"{kx_phys:12.6f} {ky_phys:12.6f} {a:3d} {b:3d} {mean_val.real:15.8e} {mean_val.imag:15.8e} {error_val.real:15.8e} {error_val.imag:15.8e}\n")
                
            except Exception as e:
                print(f"Error analyzing k-space equal-time observable {obs_name}: {e}")
    
    # Analyze unequal-time observables (both real space and k-space)
    if uneqtime_data_r and uneqtime_data_k and lattice_info:
        # Process real space unequal-time observables
        for obs_name, data in uneqtime_data_r.items():
            try:
                mean, error = jackknife_analysis_array(data)
                
                # Create directory for this observable if it doesn't exist
                obs_dir = obs_name
                if not os.path.exists(obs_dir):
                    os.makedirs(obs_dir)
                
                # Write real space data to statr.dat
                statr_file = os.path.join(obs_dir, "statr.dat")
                with open(statr_file, 'w') as f:
                    f.write(f"# Unequal-time observable: {obs_name} (Real space)\n")
                    f.write(f"# Dimensions: {mean.shape}\n")
                    f.write("# Format: rx ry a b tau mean error\n")
                    
                    # Assuming mean and error have shape (nx, ny, n_flattened) where n_flattened = (norb*norb)*ntau
                    nx, ny, n_flattened = mean.shape
                    norb = lattice_info['n_orb']
                    ntau = n_flattened // (norb * norb)  # Calculate ntau from flattened dimension
                    
                    for x in range(nx):
                        for y in range(ny):
                            for flat_idx in range(n_flattened):
                                # Convert flattened index to a, b, tau indices
                                # dim3 = (a*n_orb + b)*n_tau + tau
                                tau = flat_idx % ntau
                                orb_ab = flat_idx // ntau
                                b = orb_ab % norb
                                a = orb_ab // norb
                                
                                # Convert indices to physical positions
                                rx, ry = convert_r_indices_to_physical(x, y, lattice_info)
                                f.write(f"{rx:12.6f} {ry:12.6f} {a:3d} {b:3d} {tau:3d} {mean[x, y, flat_idx]:15.8e} {error[x, y, flat_idx]:15.8e}\n")
                
                # Write real space averaged data to statr0.dat (data at rx=0, ry=0)
                statr0_file = os.path.join(obs_dir, "statr0.dat")
                with open(statr0_file, 'w') as f:
                    f.write(f"# Unequal-time observable: {obs_name} (Real space, at rx=0, ry=0)\n")
                    f.write(f"# Dimensions: {mean.shape}\n")
                    f.write("# Format: a b tau mean error\n")
                    
                    # Find the indices corresponding to rx=0, ry=0
                    # For a square lattice with PBC, rx=0, ry=0 corresponds to x=Lx/2-1, y=Ly/2-1
                    L1 = lattice_info['L1']
                    L2 = lattice_info['L2']
                    x0 = L1 // 2 - 1
                    y0 = L2 // 2 - 1
                    
                    # Make sure indices are within bounds
                    x0 = max(0, min(x0, L1 - 1))
                    y0 = max(0, min(y0, L2 - 1))
                    
                    # Extract data at rx=0, ry=0
                    mean_r0 = mean[x0, y0, :]
                    error_r0 = error[x0, y0, :]
                    
                    for flat_idx in range(n_flattened):
                        # Convert flattened index to a, b, tau indices
                        # dim3 = (a*n_orb + b)*n_tau + tau
                        tau = flat_idx % ntau
                        orb_ab = flat_idx // ntau
                        b = orb_ab % norb
                        a = orb_ab // norb
                        
                        f.write(f"{a:3d} {b:3d} {tau:3d} {mean_r0[flat_idx]:15.8e} {error_r0[flat_idx]:15.8e}\n")
                
            except Exception as e:
                print(f"Error analyzing real-space unequal-time observable {obs_name}: {e}")
        
        # Process k-space unequal-time observables
        for obs_name, data in uneqtime_data_k.items():
            try:
                # For complex data, we need to handle real and imaginary parts
                # Convert to complex array first
                # Data is stored as [nkx, nky, n_flattened, 2] where last dimension is [real, imag]
                complex_data = []
                for d in data:
                    real_part = d[:, :, :, 0]
                    imag_part = d[:, :, :, 1]
                    complex_d = real_part + 1j * imag_part
                    complex_data.append(complex_d)
                
                mean, error = jackknife_analysis_array(complex_data)
                
                # Create directory for this observable if it doesn't exist
                obs_dir = obs_name
                if not os.path.exists(obs_dir):
                    os.makedirs(obs_dir)
                
                # Write k-space data to statk.dat
                statk_file = os.path.join(obs_dir, "statk.dat")
                with open(statk_file, 'w') as f:
                    f.write(f"# Unequal-time observable: {obs_name} (K-space)\n")
                    f.write(f"# Dimensions: {mean.shape}\n")
                    f.write("# Format: kx ky a b tau mean_real mean_imag error_real error_imag\n")
                    
                    # Assuming mean and error have shape (nkx, nky, n_flattened) where n_flattened = (norb*norb)*ntau
                    nkx, nky, n_flattened = mean.shape
                    norb = lattice_info['n_orb']
                    ntau = n_flattened // (norb * norb)  # Calculate ntau from flattened dimension
                    
                    for kx in range(nkx):
                        for ky in range(nky):
                            for flat_idx in range(n_flattened):
                                # Convert flattened index to a, b, tau indices
                                # dim3 = (a*n_orb + b)*n_tau + tau
                                tau = flat_idx % ntau
                                orb_ab = flat_idx // ntau
                                b = orb_ab % norb
                                a = orb_ab // norb
                                
                                # Convert indices to physical k-vectors
                                kx_phys, ky_phys = convert_k_indices_to_physical(kx, ky, lattice_info)
                                mean_val = mean[kx, ky, flat_idx]
                                error_val = error[kx, ky, flat_idx]
                                f.write(f"{kx_phys:12.6f} {ky_phys:12.6f} {a:3d} {b:3d} {tau:3d} {mean_val.real:15.8e} {mean_val.imag:15.8e} {error_val.real:15.8e} {error_val.imag:15.8e}\n")
                
            except Exception as e:
                print(f"Error analyzing k-space unequal-time observable {obs_name}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Jackknife analysis for DQMC observables")
    parser.add_argument("-d", "--directory", default="results", 
                        help="Results directory (default: results)")
    
    args = parser.parse_args()
    
    try:
        # Check if Parallel Tempering is enabled by reading parameters.in
        pt_enabled = is_pt_enabled("parameters.in")

        # Load lattice information
        lattice_info = load_lattice_info(args.directory)
        
        # Load data based on whether PT is enabled
        scalar_data = load_scalar_data(args.directory, pt_enabled=pt_enabled)
        eqtime_data_r, eqtime_data_k = load_equaltime_data(args.directory, pt_enabled=pt_enabled)
        uneqtime_data_r, uneqtime_data_k = load_unequaltime_data(args.directory, pt_enabled=pt_enabled)
        
        if not scalar_data and not eqtime_data_r and not eqtime_data_k and not uneqtime_data_r:
            print("No observables found in data files")
            return
        
        # Perform analysis
        results = analyze_observables(scalar_data, eqtime_data_r, eqtime_data_k, 
                                    uneqtime_data_r, uneqtime_data_k, lattice_info)
        
        # Print summary
        total_measurements = 0
        if scalar_data:
            total_measurements = len(next(iter(scalar_data.values())))
            print(f"Total measurements: {total_measurements}")
        
        # Print success messages for each observable
        all_observables = set()
        if scalar_data:
            all_observables.update(scalar_data.keys())
        if eqtime_data_r:
            all_observables.update(eqtime_data_r.keys())
        if uneqtime_data_r:
            all_observables.update(uneqtime_data_r.keys())
            
        for obs_name in sorted(all_observables):
            print(f"{obs_name} success.")
        
        print("Analysis complete.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return


if __name__ == "__main__":
    main()