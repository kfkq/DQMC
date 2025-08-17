// include/mutuner.hpp

#ifndef MUTUNER_HPP
#define MUTUNER_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <stdexcept>

class MuTuner {
private:
    // --- Parameters ---
    double n0_;      // Target density (intensive)
    double N0_;      // Target particle number (extensive)
    double beta_;    // Inverse temperature
    int V_;          // System volume (number of sites)
    double u0_;      // Characteristic energy scale
    double alpha_;   // Extensive energy scale (V / u0)
    double c_;       // Fraction of history to KEEP (1.0 - memory_fraction)
    int fixed_window_size_; // Size of the fixed window (0 means disabled)
    double tolerance_; // Convergence tolerance
    int min_sweeps_; // Minimum sweeps before checking convergence

    // --- State Variables ---
    int t_ = -1;     // Current time step, starts at -1, first update is t=0
    double mu_tp1_;  // Chemical potential for the next step (t+1)
    double mu_bar_ = 0.0;
    double mu_var_ = 0.0;
    double N_bar_ = 0.0;
    double N_var_ = 0.0;
    double N2_bar_ = 0.0;
    double kappa_bar_ = 0.0;

    // --- History ---
    std::vector<double> mu_traj_;
    std::vector<double> N_traj_;
    std::vector<double> N2_traj_;

public:
    // --- CONSTRUCTOR (MODIFIED) ---
    // Now accepts `fixed_window_size` to select the averaging mode.
    MuTuner(double target_density, double beta, int system_size, double energy_scale, 
            double initial_mu, double memory_fraction, int fixed_window_size,
            double tolerance, int min_sweeps)
        : n0_(target_density), beta_(beta), V_(system_size), u0_(energy_scale), 
          c_(1.0 - memory_fraction),
          fixed_window_size_(fixed_window_size),
          tolerance_(tolerance), min_sweeps_(min_sweeps)
    {
        if (fixed_window_size_ < 0) {
            throw std::invalid_argument("fixed_window_size cannot be negative.");
        }

        N0_ = n0_ * V_;
        alpha_ = static_cast<double>(V_) / u0_;
        mu_tp1_ = initial_mu;
        mu_bar_ = initial_mu;
        kappa_bar_ = alpha_; // Initial guess for compressibility
    }

    // --- UPDATE FUNCTION (MODIFIED) ---
    // Takes density and N^2 for clarity and consistency with your observables code.
    double update(double current_N, double current_N_squared) {
        t_++; // Increment time step
        
        double N = current_N;

        // Record history
        mu_traj_.push_back(mu_tp1_);
        N_traj_.push_back(N);
        N2_traj_.push_back(current_N_squared);

        // Update means and variances using the chosen windowing method
        update_windowed_mv();

        // --- Calculate bounded compressibility (kappa) ---
        N_var_ = N2_bar_ - (N_bar_ * N_bar_);
        if (N_var_ < 0) N_var_ = 0; // Ensure variance is not negative due to floating point errors

        // 1. Fluctuation-based estimator
        double kappa_fluc = beta_ * N_var_;

        // 2. Lower bound
        double kappa_min = alpha_ / std::sqrt(static_cast<double>(t_ + 1));

        // 3. Upper bound
        double kappa_max = kappa_min; // Default safety value
        if (t_ > 0 && mu_var_ > 1e-12 && (N_var_ / mu_var_) >= 0) {
            kappa_max = std::sqrt(N_var_ / mu_var_);
        }

        // Combine into the final bounded estimator
        kappa_bar_ = std::max(kappa_min, std::min(kappa_max, kappa_fluc));

        // --- Update the chemical potential ---
        if (std::isfinite(kappa_bar_) && kappa_bar_ > 1e-12) {
            mu_tp1_ = mu_bar_ + (N0_ - N_bar_) / kappa_bar_;
        } else {
            mu_tp1_ = mu_bar_; // Fallback if kappa is unstable
        }

        return mu_tp1_;
    }

    bool is_converged() const {
        if (t_ < min_sweeps_) {
            return false;
        }
        double error = std::abs(N_bar_ - N0_) / V_;
        return error < tolerance_;
    }

    void print_status() const {
        if (V_ == 0) return; // Avoid division by zero
        double current_density = N_bar_ / V_;
        double error = std::abs(current_density - n0_);
        std::cout << "  [MuTuner] Sweep: " << std::setw(5) << t_ + 1
                  << " | mu: " << std::fixed << std::setprecision(6) << mu_tp1_
                  << " | <n>: " << std::fixed << std::setprecision(6) << current_density
                  << " | Error: " << std::scientific << std::setprecision(4) << error
                  << std::endl;
    }

private:
    // --- HELPER FUNCTIONS (REWRITTEN FOR CLARITY AND CORRECTNESS) ---

    // Master function to calculate all windowed averages.
    void update_windowed_mv() {
        mu_bar_ = calculate_windowed_mean(mu_traj_);
        mu_var_ = calculate_windowed_var(mu_traj_, mu_bar_);

        N_bar_ = calculate_windowed_mean(N_traj_);
        N2_bar_ = calculate_windowed_mean(N2_traj_);
    }

    // Calculates the mean, automatically handling both fixed and growing windows.
    double calculate_windowed_mean(const std::vector<double>& x) const {
        if (x.empty()) return 0.0;

        int start_index = 0;
        int num_points_in_window = x.size();

        if (fixed_window_size_ > 0) {
            // --- FIXED-SIZE SLIDING WINDOW LOGIC ---
            if (num_points_in_window > fixed_window_size_) {
                // The window is full, so we slide it.
                start_index = num_points_in_window - fixed_window_size_;
                num_points_in_window = fixed_window_size_;
            }
            // **FIX**: If the window is not full yet, we correctly average over all
            // available points (start_index remains 0, num_points_in_window is x.size()).
        } else {
            // --- ORIGINAL GROWING WINDOW LOGIC ---
            start_index = static_cast<int>(std::ceil((1.0 - c_) * (x.size() - 1)));
            num_points_in_window = x.size() - start_index;
        }
        
        if (num_points_in_window <= 0) return x.back(); // Safety for edge cases

        double sum = 0.0;
        for (int i = start_index; i < x.size(); ++i) {
            sum += x[i];
        }
        return sum / num_points_in_window;
    }

    // Calculates the variance, automatically handling both fixed and growing windows.
    double calculate_windowed_var(const std::vector<double>& x, double mean) const {
        if (x.size() < 2) return 0.0;

        int start_index = 0;
        int num_points_in_window = x.size();

        if (fixed_window_size_ > 0) {
            // --- FIXED-SIZE SLIDING WINDOW LOGIC ---
            if (num_points_in_window > fixed_window_size_) {
                start_index = num_points_in_window - fixed_window_size_;
                num_points_in_window = fixed_window_size_;
            }
        } else {
            // --- ORIGINAL GROWING WINDOW LOGIC ---
            start_index = static_cast<int>(std::ceil((1.0 - c_) * (x.size() - 1)));
            num_points_in_window = x.size() - start_index;
        }

        if (num_points_in_window <= 1) return 0.0;

        double sum_sq_diff = 0.0;
        for (int i = start_index; i < x.size(); ++i) {
            sum_sq_diff += (x[i] - mean) * (x[i] - mean);
        }
        return sum_sq_diff / num_points_in_window;
    }
};

#endif // MUTUNER_HPP