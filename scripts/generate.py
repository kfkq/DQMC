#!/usr/bin/env python3

import os
import itertools
import argparse
import textwrap

def generate_simulation_files(base_dir, params_config):
    """
    Generates the directory and files for a single simulation configuration.

    Args:
        base_dir (str): The top-level directory for the campaign.
        params_config (dict): A dictionary containing all parameters for this specific run.
    """
    # --- Handle Parallel Tempering Setup ---
    pt_config = params_config.pop('parallel_tempering', None)
    is_pt_run = isinstance(pt_config, dict) and pt_config.get('enabled', False)

    # The main 'beta' and 'mu' from the grid are now the TARGET values.
    # For PT runs, these will be the first replica's values
    target_beta = params_config['beta']
    target_mu = params_config['mu']

    # --- Create a descriptive directory name from the parameters ---
    dir_name_parts = []
    # Sort keys for consistent naming
    for key, value in sorted(params_config.items()):
        dir_name_parts.append(f"{key}_{value}")
    
    # Add a special tag for PT runs
    dir_name_parts.append(f"PT_{is_pt_run}")
    
    dir_name = "-".join(dir_name_parts).replace(" ", "")
    sim_path = os.path.join(base_dir, dir_name)
    os.makedirs(sim_path, exist_ok=True)

    # --- Prepare the [parallel_tempering] section text ---
    if is_pt_run:
        pt_section_str = textwrap.dedent(f"""
            [parallel_tempering]
            enabled = true
            exchange_freq = {pt_config['exchange_freq']}
        """).strip()
    else:
        pt_section_str = "[parallel_tempering]\nenabled = false"

    # --- Format the complete main parameters.in content ---
    main_params_content = textwrap.dedent(f"""
        [lattice]
        type = "square"
        Lx = {params_config['Lx']}
        Ly = {params_config['Ly']}

        [hubbard]
        U = {params_config['U']}
        t = {params_config['t']}
        mu = {target_mu}                       # Chemical potential for the TARGET replica

        {pt_section_str}

        [simulation]
        beta = {target_beta}                      # Inverse temperature for the TARGET replica
        nt = {params_config['nt']}
        n_therms = {params_config['n_therms']}
        n_sweeps = {params_config['n_sweeps']}
        n_bins = {params_config['n_bins']}
        n_stab = {params_config['n_stab']}
        isMeasureUnequalTime = {str(params_config['isMeasureUnequalTime']).lower()}
    """).strip()

    with open(os.path.join(sim_path, "parameters.in"), "w") as f:
        f.write(main_params_content)

    # --- Generate replica override files if this is a PT run ---
    if is_pt_run:
        replica_dir = os.path.join(sim_path, "parallel_tempering")
        os.makedirs(replica_dir, exist_ok=True)
        n_replicas = len(pt_config['betas'])
        
        print(f"  -> Generated PT run: {dir_name}")
        print(f"     ... with {n_replicas} replicas. Run with 'mpirun -np {n_replicas}'")

        # Generate override files for ranks 1, 2, ... N-1
        for i in range(1, n_replicas):
            beta_i = pt_config['betas'][i]
            mu_i = pt_config['mus'][i]
            
            replica_content = textwrap.dedent(f"""
                # Override for replica {i}
                [hubbard]
                mu = {mu_i}

                [simulation]
                beta = {beta_i}
            """).strip()
            with open(os.path.join(replica_dir, f"parameters_{i}.in"), "w") as f:
                f.write(replica_content)
    else:
        print(f"  -> Generated standard run: {dir_name}")

def main(output_dir):
    """
    Defines the parameter space and generates all simulation directories.
    """
    print(f"Generating simulation directories in '{output_dir}/'")
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # --- DEFINE YOUR PARAMETER SPACE HERE ---
    # Any parameter defined here with a list of values will be swept over.
    # A directory will be created for each unique combination.
    # =========================================================================
    parameter_space = {
        # Lattice
        'Lx': [8],
        'Ly': [8],
        # Hubbard
        'U': [6.0],
        't': [1.0],
        'mu': [-0.4], # This is the TARGET mu
        # Simulation
        'beta': [5.0], # This is the TARGET beta
        'nt': [100],
        'n_therms': [0],
        'n_sweeps': [50],
        'n_bins': [500],
        'n_stab': [5],
        'isMeasureUnequalTime': [False],
        # Parallel Tempering (Special Parameter)
        # This parameter expects a list of configurations. Use `None` for a standard run.
        'parallel_tempering': [
            # None, # A standard run without PT
            {      # A PT configuration
                "enabled": True,
                "exchange_freq": 10,
                "betas": [5.0, 5.0, 2.0, 1.0], # Target (rank 0) is first
                "mus":   [-0.4, -0.4, -0.48, -0.71] # Target (rank 0) is first
            }
        ]
    }
    # =========================================================================

    # --- Generate Cartesian product of the parameter space ---
    keys, values = zip(*parameter_space.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # --- Loop through all combinations to generate the files ---
    for params in all_combinations:
        generate_simulation_files(output_dir, params)
            
    print("\nGeneration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate directory structures for DQMC simulations.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="simulations",
        help="The base directory to generate the simulation folders in. (Default: 'simulations')"
    )
    args = parser.parse_args()
    
    main(output_dir=args.output_dir)
