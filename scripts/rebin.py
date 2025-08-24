#!/usr/bin/env python3
"""
Script to average binned data over MPI ranks and save to a single HDF5 file.
Averages per bin_i across all data_*.h5 files and saves to data.h5.
"""

import h5py
import numpy as np
import os
import glob
import argparse


def average_bins(results_dir="results"):
    """
    Average data per bin across all MPI rank files and save to data.h5.
    
    Args:
        results_dir (str): Path to the results directory
    """
    # Find all data files
    data_files = glob.glob(os.path.join(results_dir, "data_*.h5"))
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {results_dir}")
    
    # Open the first file to get bin groups
    with h5py.File(data_files[0], 'r') as f_first:
        bin_groups = [key for key in f_first.keys() if key.startswith('bin_') and not key.startswith('binK_')]
        bin_groups.sort(key=lambda x: int(x.split('_')[1]))
        
        k_bin_groups = [key for key in f_first.keys() if key.startswith('binK_')]
        k_bin_groups.sort(key=lambda x: int(x.split('_')[1]))
    
    all_bin_groups = bin_groups + k_bin_groups
    
    if not all_bin_groups:
        print("No bin groups found in data files.")
        return
    
    # Create or overwrite data.h5
    output_path = os.path.join(results_dir, "data.h5")
    with h5py.File(output_path, 'w') as f_out:
        for bin_name in all_bin_groups:
            # Create the bin group in output
            bin_group_out = f_out.create_group(bin_name)
            
            # Determine possible subgroups (scalar, equaltime, unequaltime)
            subgroups = ['scalar', 'equaltime', 'unequaltime']
            
            for subgroup_name in subgroups:
                obs_collected = {}  # obs_name -> list of data from ranks
                
                # Collect data from all files for this bin/subgroup
                for file_path in data_files:
                    with h5py.File(file_path, 'r') as f_in:
                        if bin_name in f_in and subgroup_name in f_in[bin_name]:
                            subgroup_in = f_in[bin_name][subgroup_name]
                            
                            for obs_name in subgroup_in.keys():
                                if obs_name not in obs_collected:
                                    obs_collected[obs_name] = []
                                
                                dataset = subgroup_in[obs_name]
                                value = np.array(dataset)
                                
                                # For scalars: handle 0D or 1D single-element
                                if value.ndim == 0:
                                    value = np.array([value])
                                elif value.ndim == 1 and value.size == 1:
                                    pass  # Already fine
                                
                                obs_collected[obs_name].append(value)
                
                # If any observables collected for this subgroup, average and save
                if obs_collected:
                    subgroup_out = bin_group_out.create_group(subgroup_name)
                    
                    for obs_name, data_list in obs_collected.items():
                        if not data_list:
                            continue
                        
                        # Stack along new axis (axis=0)
                        stacked = np.stack(data_list, axis=0)
                        
                        # Average over ranks
                        averaged = np.mean(stacked, axis=0)
                        
                        # For scalars, squeeze if possible
                        if averaged.size == 1:
                            averaged = averaged.reshape(())
                        
                        # Save as dataset
                        subgroup_out.create_dataset(obs_name, data=averaged)
    
    print(f"Averaging complete. Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Average DQMC bins over MPI ranks")
    parser.add_argument("-d", "--directory", default="results", 
                        help="Results directory (default: results)")
    
    args = parser.parse_args()
    
    try:
        average_bins(args.directory)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()