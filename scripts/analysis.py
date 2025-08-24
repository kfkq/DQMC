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
from pathlib import Path


def load_scalar_data(results_dir="results", discard_bins=0):
    data_file = os.path.join(results_dir, "data.h5")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found")
    
    scalar_data = {}
    
    with h5py.File(data_file, 'r') as f:
        bin_groups = [key for key in f.keys() if key.startswith('bin_') and not key.startswith('binK_')]
        bin_groups.sort(key=lambda x: int(x.split('_')[1]))
        
        bin_groups = bin_groups[discard_bins:]
        
        for bin_group in bin_groups:
            group = f[bin_group]
            
            if 'scalar' in group:
                scalar_group = group['scalar']
                
                for obs_name in scalar_group.keys():
                    dataset = scalar_group[obs_name]
                    if len(dataset.shape) == 0 or (len(dataset.shape) == 1 and dataset.shape[0] == 1):
                        if len(dataset.shape) == 0:
                            value = dataset[()]
                        else:
                            value = dataset[0]
                        
                        if obs_name not in scalar_data:
                            scalar_data[obs_name] = []
                        
                        scalar_data[obs_name].append(value)
    
    for obs_name in scalar_data:
        scalar_data[obs_name] = np.array(scalar_data[obs_name])
    
    return scalar_data


def load_equaltime_data(results_dir="results", discard_bins=0):
    data_file = os.path.join(results_dir, "data.h5")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found")
    
    real_space_data = {}
    k_space_data = {}
    
    with h5py.File(data_file, 'r') as f:
        bin_groups = [key for key in f.keys() if key.startswith('bin_') and not key.startswith('binK_')]
        bin_groups.sort(key=lambda x: int(x.split('_')[1]))
        bin_groups = bin_groups[discard_bins:]
        
        k_bin_groups = [key for key in f.keys() if key.startswith('binK_')]
        k_bin_groups.sort(key=lambda x: int(x.split('_')[1]))
        k_bin_groups = k_bin_groups[discard_bins:]
        
        for bin_group in bin_groups:
            group = f[bin_group]
            
            if 'equaltime' in group:
                eqtime_group = group['equaltime']
                
                for obs_name in eqtime_group.keys():
                    dataset = eqtime_group[obs_name]
                    value = np.array(dataset)
                    
                    if obs_name not in real_space_data:
                        real_space_data[obs_name] = []
                    
                    real_space_data[obs_name].append(value)
        
        for bin_group in k_bin_groups:
            group = f[bin_group]
            
            if 'equaltime' in group:
                eqtime_group = group['equaltime']
                
                for obs_name in eqtime_group.keys():
                    dataset = eqtime_group[obs_name]
                    value = np.array(dataset)
                    
                    if obs_name not in k_space_data:
                        k_space_data[obs_name] = []
                    
                    k_space_data[obs_name].append(value)
    
    return real_space_data, k_space_data


def load_unequaltime_data(results_dir="results", discard_bins=0):
    data_file = os.path.join(results_dir, "data.h5")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found")
    
    real_space_data = {}
    k_space_data = {}
    
    with h5py.File(data_file, 'r') as f:
        bin_groups = [key for key in f.keys() if key.startswith('bin_') and not key.startswith('binK_')]
        bin_groups.sort(key=lambda x: int(x.split('_')[1]))
        bin_groups = bin_groups[discard_bins:]
        
        k_bin_groups = [key for key in f.keys() if key.startswith('binK_')]
        k_bin_groups.sort(key=lambda x: int(x.split('_')[1]))
        k_bin_groups = k_bin_groups[discard_bins:]
        
        for bin_group in bin_groups:
            group = f[bin_group]
            
            if 'unequaltime' in group:
                uneqtime_group = group['unequaltime']
                
                for obs_name in uneqtime_group.keys():
                    dataset = uneqtime_group[obs_name]
                    value = np.array(dataset)
                    
                    if obs_name not in real_space_data:
                        real_space_data[obs_name] = []
                    
                    real_space_data[obs_name].append(value)
        
        for bin_group in k_bin_groups:
            group = f[bin_group]
            
            if 'unequaltime' in group:
                uneqtime_group = group['unequaltime']
                
                for obs_name in uneqtime_group.keys():
                    dataset = uneqtime_group[obs_name]
                    value = np.array(dataset)
                    
                    if obs_name not in k_space_data:
                        k_space_data[obs_name] = []
                    
                    k_space_data[obs_name].append(value)
    
    return real_space_data, k_space_data


def jackknife_analysis(data):
    n_bins = len(data)
    if n_bins < 2:
        raise ValueError("Need at least 2 bins for jackknife analysis")
    
    full_mean = np.mean(data)
    
    jackknife_means = np.zeros(n_bins)
    for i in range(n_bins):
        jackknife_means[i] = (n_bins * full_mean - data[i]) / (n_bins - 1)
    
    jackknife_mean = np.mean(jackknife_means)
    variance = np.sum((jackknife_means - jackknife_mean)**2) * (n_bins - 1) / n_bins
    error = np.sqrt(variance)
    
    return full_mean, error


def jackknife_analysis_array(data):
    n_bins = len(data)
    if n_bins < 2:
        raise ValueError("Need at least 2 bins for jackknife analysis")
    
    data_array = np.array(data)
    
    full_mean = np.mean(data_array, axis=0)
    
    jackknife_means = np.zeros((n_bins,) + data_array.shape[1:], dtype=data_array.dtype)
    for i in range(n_bins):
        jackknife_means[i] = (n_bins * full_mean - data_array[i]) / (n_bins - 1)
    
    jackknife_mean = np.mean(jackknife_means, axis=0)
    variance = np.sum((jackknife_means - jackknife_mean[np.newaxis, ...])**2, axis=0) * (n_bins - 1) / n_bins
    error = np.sqrt(variance)
    
    return full_mean, error


def load_lattice_info(results_dir="results"):
    info_file = os.path.join(results_dir, "info")
    lattice_info = {}
    
    try:
        with open(info_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    lattice_info[key] = value
    except FileNotFoundError:
        print(f"Error: Lattice info file {info_file} not found.")
        print("This file is required for coordinate conversion.")
        print("Please run the simulation first to generate this file.")
        exit(1)
    
    return lattice_info


def convert_r_indices_to_physical(x_idx, y_idx, lattice_info):
    Lx = lattice_info['Lx']
    Ly = lattice_info['Ly']
    a1_x = lattice_info['a1_x']
    a1_y = lattice_info['a1_y']
    a2_x = lattice_info['a2_x']
    a2_y = lattice_info['a2_y']
    
    dx_phys = (x_idx - (Lx/2 - 1)) * a1_x + (y_idx - (Ly/2 - 1)) * a2_x
    dy_phys = (x_idx - (Lx/2 - 1)) * a1_y + (y_idx - (Ly/2 - 1)) * a2_y
    
    return dx_phys, dy_phys


def convert_k_indices_to_physical(kx_idx, ky_idx, lattice_info):
    Lx = lattice_info['Lx']
    Ly = lattice_info['Ly']
    
    a1_x = lattice_info['a1_x']
    a1_y = lattice_info['a1_y']
    a2_x = lattice_info['a2_x']
    a2_y = lattice_info['a2_y']
    
    det = a1_x * a2_y - a1_y * a2_x
    
    b1_x = 2 * np.pi * a2_y / det / Lx
    b1_y = -2 * np.pi * a2_x / det / Lx
    b2_x = -2 * np.pi * a1_y / det / Ly
    b2_y = 2 * np.pi * a1_x / det / Ly
    
    qx = kx_idx - (Lx // 2) + 1
    qy = ky_idx - (Ly // 2) + 1
    
    kx_phys = qx * b1_x + qy * b2_x
    ky_phys = qx * b1_y + qy * b2_y
    
    return kx_phys, ky_phys


def analyze_observables(scalar_data, eqtime_data_r=None, eqtime_data_k=None, 
                       uneqtime_data_r=None, uneqtime_data_k=None, lattice_info=None):
    results = {}
    
    for obs_name, data in scalar_data.items():
        try:
            mean, error = jackknife_analysis(data)
            results[obs_name] = (mean, error)
        except Exception as e:
            print(f"Error analyzing scalar observable {obs_name}: {e}")
            results[obs_name] = (np.nan, np.nan)
    
    output_filename = "scalarObservables.dat"
    if results:
        with open(output_filename, 'w') as f:
            f.write("# Observable Mean Error\n")
            for obs_name, (mean, error) in results.items():
                f.write(f"{obs_name} {mean} {error}\n")
    
    if eqtime_data_r and eqtime_data_k and lattice_info:
        for obs_name, data in eqtime_data_r.items():
            try:
                mean, error = jackknife_analysis_array(data)
                
                obs_dir = obs_name
                if not os.path.exists(obs_dir):
                    os.makedirs(obs_dir)
                
                statr_file = os.path.join(obs_dir, "statr.dat")
                with open(statr_file, 'w') as f:
                    f.write(f"# Equal-time observable: {obs_name} (Real space)\n")
                    f.write(f"# Dimensions: {mean.shape}\n")
                    f.write("# Format: rx ry a b mean error\n")
                    
                    nx, ny, norb = mean.shape
                    
                    for x in range(nx):
                        for y in range(ny):
                            for orb_idx in range(norb):
                                a = orb_idx // lattice_info['n_orb'] if 'n_orb' in lattice_info else 0
                                b = orb_idx % lattice_info['n_orb'] if 'n_orb' in lattice_info else orb_idx
                                
                                rx, ry = convert_r_indices_to_physical(x, y, lattice_info)
                                f.write(f"{rx:12.6f} {ry:12.6f} {a:3d} {b:3d} {mean[x, y, orb_idx]:15.8e} {error[x, y, orb_idx]:15.8e}\n")
                
            except Exception as e:
                print(f"Error analyzing real-space equal-time observable {obs_name}: {e}")
        
        for obs_name, data in eqtime_data_k.items():
            try:
                complex_data = []
                for d in data:
                    real_part = d[:, :, :, 0]
                    imag_part = d[:, :, :, 1]
                    complex_d = real_part + 1j * imag_part
                    complex_data.append(complex_d)
                
                mean, error = jackknife_analysis_array(complex_data)
                
                obs_dir = obs_name
                if not os.path.exists(obs_dir):
                    os.makedirs(obs_dir)
                
                statk_file = os.path.join(obs_dir, "statk.dat")
                with open(statk_file, 'w') as f:
                    f.write(f"# Equal-time observable: {obs_name} (K-space)\n")
                    f.write(f"# Dimensions: {mean.shape}\n")
                    f.write("# Format: kx ky a b mean_real mean_imag error_real error_imag\n")
                    
                    nkx, nky, norb = mean.shape
                    
                    for kx in range(nkx):
                        for ky in range(nky):
                            for orb_idx in range(norb):
                                a = orb_idx // lattice_info['n_orb'] if 'n_orb' in lattice_info else 0
                                b = orb_idx % lattice_info['n_orb'] if 'n_orb' in lattice_info else orb_idx
                                
                                kx_phys, ky_phys = convert_k_indices_to_physical(kx, ky, lattice_info)
                                mean_val = mean[kx, ky, orb_idx]
                                error_val = error[kx, ky, orb_idx]
                                f.write(f"{kx_phys:12.6f} {ky_phys:12.6f} {a:3d} {b:3d} {mean_val.real:15.8e} {mean_val.imag:15.8e} {error_val.real:15.8e} {error_val.imag:15.8e}\n")
                
            except Exception as e:
                print(f"Error analyzing k-space equal-time observable {obs_name}: {e}")
    
    if uneqtime_data_r and uneqtime_data_k and lattice_info:
        for obs_name, data in uneqtime_data_r.items():
            try:
                mean, error = jackknife_analysis_array(data)
                
                obs_dir = obs_name
                if not os.path.exists(obs_dir):
                    os.makedirs(obs_dir)
                
                statr_file = os.path.join(obs_dir, "statr.dat")
                with open(statr_file, 'w') as f:
                    f.write(f"# Unequal-time observable: {obs_name} (Real space)\n")
                    f.write(f"# Dimensions: {mean.shape}\n")
                    f.write("# Format: rx ry a b tau mean error\n")
                    
                    nx, ny, n_flattened = mean.shape
                    norb = lattice_info['n_orb']
                    ntau = n_flattened // (norb * norb)
                    
                    for x in range(nx):
                        for y in range(ny):
                            for flat_idx in range(n_flattened):
                                tau = flat_idx % ntau
                                orb_ab = flat_idx // ntau
                                b = orb_ab % norb
                                a = orb_ab // norb
                                
                                rx, ry = convert_r_indices_to_physical(x, y, lattice_info)
                                f.write(f"{rx:12.6f} {ry:12.6f} {a:3d} {b:3d} {tau:3d} {mean[x, y, flat_idx]:15.8e} {error[x, y, flat_idx]:15.8e}\n")
                
                statr0_file = os.path.join(obs_dir, "statr0.dat")
                with open(statr0_file, 'w') as f:
                    f.write(f"# Unequal-time observable: {obs_name} (Real space, at rx=0, ry=0)\n")
                    f.write(f"# Dimensions: {mean.shape}\n")
                    f.write("# Format: a b tau mean error\n")
                    
                    Lx = lattice_info['Lx']
                    Ly = lattice_info['Ly']
                    x0 = Lx // 2 - 1
                    y0 = Ly // 2 - 1
                    
                    x0 = max(0, min(x0, Lx - 1))
                    y0 = max(0, min(y0, Ly - 1))
                    
                    mean_r0 = mean[x0, y0, :]
                    error_r0 = error[x0, y0, :]
                    
                    for flat_idx in range(n_flattened):
                        tau = flat_idx % ntau
                        orb_ab = flat_idx // ntau
                        b = orb_ab % norb
                        a = orb_ab // norb
                        
                        f.write(f"{a:3d} {b:3d} {tau:3d} {mean_r0[flat_idx]:15.8e} {error_r0[flat_idx]:15.8e}\n")
                
            except Exception as e:
                print(f"Error analyzing real-space unequal-time observable {obs_name}: {e}")
        
        for obs_name, data in uneqtime_data_k.items():
            try:
                complex_data = []
                for d in data:
                    real_part = d[:, :, :, 0]
                    imag_part = d[:, :, :, 1]
                    complex_d = real_part + 1j * imag_part
                    complex_data.append(complex_d)
                
                mean, error = jackknife_analysis_array(complex_data)
                
                obs_dir = obs_name
                if not os.path.exists(obs_dir):
                    os.makedirs(obs_dir)
                
                statk_file = os.path.join(obs_dir, "statk.dat")
                with open(statk_file, 'w') as f:
                    f.write(f"# Unequal-time observable: {obs_name} (K-space)\n")
                    f.write(f"# Dimensions: {mean.shape}\n")
                    f.write("# Format: kx ky a b tau mean_real mean_imag error_real error_imag\n")
                    
                    nkx, nky, n_flattened = mean.shape
                    norb = lattice_info['n_orb']
                    ntau = n_flattened // (norb * norb)
                    
                    for kx in range(nkx):
                        for ky in range(nky):
                            for flat_idx in range(n_flattened):
                                tau = flat_idx % ntau
                                orb_ab = flat_idx // ntau
                                b = orb_ab % norb
                                a = orb_ab // norb
                                
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
    parser.add_argument("discard_bins", type=int, nargs='?', default=0,
                        help="Number of initial bins to discard (default: 0)")
    
    args = parser.parse_args()
    
    try:
        lattice_info = load_lattice_info(args.directory)
        
        scalar_data = load_scalar_data(args.directory, args.discard_bins)
        
        eqtime_data_r, eqtime_data_k = load_equaltime_data(args.directory, args.discard_bins)
        
        uneqtime_data_r, uneqtime_data_k = load_unequaltime_data(args.directory, args.discard_bins)
        
        if not scalar_data and not eqtime_data_r and not eqtime_data_k and not uneqtime_data_r:
            print("No observables found in data files")
            return
        
        results = analyze_observables(scalar_data, eqtime_data_r, eqtime_data_k, 
                                    uneqtime_data_r, uneqtime_data_k, lattice_info)
        
        total_measurements = 0
        if scalar_data:
            total_measurements = len(next(iter(scalar_data.values())))
            print(f"Total measurements: {total_measurements}")
        
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