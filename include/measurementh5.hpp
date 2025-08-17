#ifndef MEASUREMENT_HPP
#define MEASUREMENT_HPP

#include "dqmc.hpp"
#include "h5utils.hpp"
#include "utility.hpp"
#include "observables.hpp"

#include <mpi.h>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <map>
#include <tuple>
#include <set>
#include <hdf5.h>
#include <memory>
#include <sys/stat.h>
#include <unistd.h>

namespace transform {
    inline int pbc_shortest(int d, int L) {
        if (d >  L/2)  d -= L;
        if (d <= -L/2) d += L;
        return d;
    }
    
    // Simplified version that returns an arma::cube instead of DataRow vector
    inline arma::cube chi_site_to_chi_r(const arma::cube& chi_site, const Lattice& lat) {
        const int n_orb = lat.n_orb();
        const int Lx = lat.Lx();
        const int Ly = lat.Ly();
        const int n_cells = lat.size();
        const int n_tau = chi_site.n_slices;

        // Determine the size of the output cube
        // For simplicity, we'll use a fixed size based on Lx and Ly
        const int dx_size = Lx;
        const int dy_size = Ly;
        const int orb_size = n_orb * n_orb;
        
        // Create output cube
        arma::cube result(dx_size, dy_size, n_tau * orb_size);
        result.zeros();

        for (int tau = 0; tau < n_tau; ++tau) {
            // Get a view of the current tau-slice
            arma::mat chi_slice = chi_site.slice(tau);

            for (int ij = 0; ij < static_cast<int>(chi_slice.n_elem); ++ij) {
                const int i = ij % chi_slice.n_rows;
                const int j = ij / chi_slice.n_rows;
                const double val = chi_slice(i,j);
                const int a = i % n_orb;
                const int b = j % n_orb;
                const int cell_i = i / n_orb;
                const int cell_j = j / n_orb;
                const int cxi = cell_i % Lx;
                const int cyi = cell_i / Lx;
                const int cxj = cell_j % Lx;
                const int cyj = cell_j / Lx;

                int raw_dx = cxj - cxi;
                int dx_pbc = transform::pbc_shortest(raw_dx, Lx);
                int raw_dy = cyj - cyi;
                int dy_pbc = transform::pbc_shortest(raw_dy, Ly);

                int dx_idx = dx_pbc + Lx/2 - 1;
                int dy_idx = dy_pbc + Ly/2 - 1;
                
                // flattening a, b index to tau index.
                result(dx_idx, dy_idx, (a*n_orb + b)*n_tau + tau) += val / n_cells;
            }
        }
        
        return result;
    }

    // Overload for equal-time case (arma::mat)
    inline arma::cube chi_site_to_chi_r(const arma::mat& chi_site, const Lattice& lat) {
        // Wrap the matrix in a 1-slice cube, transform, and return the first slice
        arma::cube temp_cube(chi_site.n_rows, chi_site.n_cols, 1);
        temp_cube.slice(0) = chi_site;
        arma::cube result = chi_site_to_chi_r(temp_cube, lat);
        
        return result;
    }

    inline arma::cx_cube chi_r_to_chi_k(const arma::cube& chi_r, const Lattice& lat)
    {
        const auto& kpts = lat.k_points();
        const int nk = static_cast<int>(kpts.size());
        const int nt = chi_r.n_slices;
        const int nx = chi_r.n_rows;
        const int ny = chi_r.n_cols;
        const int n_orb = lat.n_orb();

        const int Lx = lat.Lx();
        const int Ly = lat.Ly();
        const auto& a1 = lat.a1();
        const auto& a2 = lat.a2();

        // Initialize complex cube with proper dimensions
        arma::cx_cube chi_k(Lx, Ly, nt);
        chi_k.zeros();

        for (int kidx = 0; kidx < nk; ++kidx) {
            const auto& k = kpts[kidx];
            int kx_idx = kidx / Lx;
            int ky_idx = kidx % Ly;
            for (int t_idx = 0; t_idx < nt; ++t_idx) {
                for (int x_idx = 0; x_idx < nx; ++x_idx) {
                    for (int y_idx = 0; y_idx < ny; ++y_idx) {
                        double dx = (x_idx - (Lx/2 - 1)) * a1[0] + (y_idx - (Ly/2 - 1)) * a2[0];
                        double dy = (x_idx - (Lx/2 - 1)) * a1[1] + (y_idx - (Ly/2 - 1)) * a2[1];

                        double phase = k[0] * dx + k[1] * dy;

                        double val = chi_r(x_idx, y_idx, t_idx);
                        chi_k(kx_idx, ky_idx, t_idx) += val * std::complex<double>(std::cos(phase), -std::sin(phase));
                    }
                }
            }
        }
        
        return chi_k;
    }
}

class MeasurementManager {
private:
    std::vector<std::function<double(const std::vector<GF>&, const Lattice&)>> scalarCalculators_;
    std::vector<std::string> scalarNames_;
    
    std::vector<std::function<arma::mat(const std::vector<GF>&, const Lattice&)>> eqTimeCalculators_;
    std::vector<std::string> eqTimeNames_;
    
    std::vector<std::function<arma::cube(const std::vector<GF>&, const Lattice&)>> uneqTimeCalculators_;
    std::vector<std::string> uneqTimeNames_;
    
    MPI_Comm comm_;
    int rank_;
    int world_size_;
    
    int current_bin_;
    
    // Storage for accumulated data
    std::vector<double> scalarData_;
    std::vector<arma::mat> eqTimeData_;
    std::vector<arma::cube> uneqTimeData_;
    
    // Accumulation counters
    int scalarCount_;
    int eqTimeCount_;
    int uneqTimeCount_;
    
    // HDF5 file handle
    hid_t file_id_;
    bool file_opened_;
    
public:
    MeasurementManager(MPI_Comm comm, int rank) 
        : comm_(comm), rank_(rank), current_bin_(0),
          scalarCount_(0), eqTimeCount_(0), uneqTimeCount_(0),
          file_opened_(false) {
        MPI_Comm_size(comm_, &world_size_);
    }
    
    ~MeasurementManager() {
        if (file_opened_) {
            hdf5::close_file(file_id_);
        }
    }
    
    void addScalar(const std::string& name, 
                   std::function<double(const std::vector<GF>&, const Lattice& lat)> calculator) {
        scalarNames_.push_back(name);
        scalarCalculators_.push_back(calculator);
        scalarData_.push_back(0.0);
    }
    
    void addEqualTime(const std::string& name,
                      std::function<arma::mat(const std::vector<GF>&, const Lattice& lat)> calculator) {
        eqTimeNames_.push_back(name);
        eqTimeCalculators_.push_back(calculator);
    }
    
    void addUnequalTime(const std::string& name,
                        std::function<arma::cube(const std::vector<GF>&, const Lattice&)> calculator) {
        uneqTimeNames_.push_back(name);
        uneqTimeCalculators_.push_back(calculator);
    }
    
    void measure(const std::vector<GF>& greens, const Lattice& lat) {
        // Measure scalar observables
        for (size_t i = 0; i < scalarCalculators_.size(); ++i) {
            scalarData_[i] += scalarCalculators_[i](greens, lat);
        }
        scalarCount_++;
        
        // Measure equal-time observables
        if (eqTimeData_.size() != eqTimeCalculators_.size()) {
            eqTimeData_.resize(eqTimeCalculators_.size());
        }
        
        for (size_t i = 0; i < eqTimeCalculators_.size(); ++i) {
            arma::mat result = eqTimeCalculators_[i](greens, lat);
            if (eqTimeData_[i].n_rows == 0) {
                eqTimeData_[i] = result;
            } else {
                eqTimeData_[i] += result;
            }
        }
        eqTimeCount_++;
    }
    
    void measure_unequalTime(const std::vector<GF>& greens, const Lattice& lat) {
        // Measure unequal-time observables
        if (uneqTimeData_.size() != uneqTimeCalculators_.size()) {
            uneqTimeData_.resize(uneqTimeCalculators_.size());
        }
        
        for (size_t i = 0; i < uneqTimeCalculators_.size(); ++i) {
            arma::cube result = uneqTimeCalculators_[i](greens, lat);
            if (uneqTimeData_[i].n_rows == 0) {
                uneqTimeData_[i] = result;
            } else {
                uneqTimeData_[i] += result;
            }
        }
        uneqTimeCount_++;
    }
    
    void accumulate(const Lattice& lat) {
        // Normalize the accumulated data
        if (scalarCount_ > 0) {
            for (size_t i = 0; i < scalarData_.size(); ++i) {
                scalarData_[i] /= scalarCount_;
            }
        }
        
        if (eqTimeCount_ > 0) {
            for (size_t i = 0; i < eqTimeData_.size(); ++i) {
                eqTimeData_[i] /= eqTimeCount_;
            }
        }
        
        if (uneqTimeCount_ > 0) {
            for (size_t i = 0; i < uneqTimeData_.size(); ++i) {
                uneqTimeData_[i] /= uneqTimeCount_;
            }
        }
        
        // Save data to HDF5 file
        saveToHDF5(lat);
        
        // Save Fourier-transformed data to HDF5 file
        saveToHDF5K(lat);
        
        // Reset accumulators
        for (size_t i = 0; i < scalarData_.size(); ++i) {
            scalarData_[i] = 0.0;
        }
        
        for (size_t i = 0; i < eqTimeData_.size(); ++i) {
            if (eqTimeData_[i].n_rows > 0) {
                eqTimeData_[i].zeros();
            }
        }
        
        for (size_t i = 0; i < uneqTimeData_.size(); ++i) {
            if (uneqTimeData_[i].n_rows > 0) {
                uneqTimeData_[i].zeros();
            }
        }
        
        scalarCount_ = 0;
        eqTimeCount_ = 0;
        uneqTimeCount_ = 0;
        
        current_bin_++;
    }
    
private:
    void saveToHDF5(const Lattice& lat) {
        // Create results directory if it doesn't exist
        struct stat info;
        if (stat("results", &info) != 0) {
            // Directory doesn't exist, create it
            #if defined(_WIN32)
            _mkdir("results");
            #else
            mkdir("results", 0755);
            #endif
        }
        // Note: HDF5 files are named with rank, so each run creates new files
        // Existing files from previous runs with same rank are overwritten by HDF5
        
        // Open file if not already opened
        if (!file_opened_) {
            // Create filename: results/data_xxx.h5 where xxx is mpi rank
            std::string filename = "results/data_" + std::to_string(rank_) + ".h5";
            file_id_ = hdf5::create_file(filename);
            file_opened_ = true;
        }
        
        // Create group for this bin
        std::string group_name = "/bin_" + std::to_string(current_bin_);
        hid_t group_id = H5Gcreate2(file_id_, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Write scalar data
        for (size_t i = 0; i < scalarNames_.size(); ++i) {
            std::string dataset_name = scalarNames_[i];
            hdf5::write_scalar(group_id, dataset_name, scalarData_[i]);
        }
        
        // Write equal-time data (raw data, no transformation)
        for (size_t i = 0; i < eqTimeNames_.size(); ++i) {
            if (eqTimeData_[i].n_rows > 0) {
                std::string dataset_name = eqTimeNames_[i];
                arma::cube chi_r = transform::chi_site_to_chi_r(eqTimeData_[i], lat);
                hdf5::write_cube(group_id, dataset_name, chi_r);
            }
        }
        
        // Write unequal-time data (raw data, no transformation)
        for (size_t i = 0; i < uneqTimeNames_.size(); ++i) {
            if (uneqTimeData_[i].n_rows > 0) {
                std::string dataset_name = uneqTimeNames_[i];
                arma::cube chi_r = transform::chi_site_to_chi_r(uneqTimeData_[i], lat);
                hdf5::write_cube(group_id, dataset_name, chi_r);
            }
        }
        
        // Close the group
        H5Gclose(group_id);
    }
    
    void saveToHDF5K(const Lattice& lat) {
        // This function should only be called after file is opened
        if (!file_opened_) return;
        
        // Create group for this bin in k-space
        std::string group_name = "/binK_" + std::to_string(current_bin_);
        hid_t group_id = H5Gcreate2(file_id_, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Write equal-time data (Fourier transformed)
        for (size_t i = 0; i < eqTimeNames_.size(); ++i) {
            if (eqTimeData_[i].n_rows > 0) {
                std::string dataset_name = eqTimeNames_[i];
                arma::cube chi_r = transform::chi_site_to_chi_r(eqTimeData_[i], lat);
                arma::cx_cube chi_k = transform::chi_r_to_chi_k(chi_r, lat);
                hdf5::write_complex_cube(group_id, dataset_name, chi_k);
            }
        }
        
        // Write unequal-time data (Fourier transformed)
        for (size_t i = 0; i < uneqTimeNames_.size(); ++i) {
            if (uneqTimeData_[i].n_rows > 0) {
                std::string dataset_name = uneqTimeNames_[i];
                arma::cube chi_r = transform::chi_site_to_chi_r(uneqTimeData_[i], lat);
                arma::cx_cube chi_k = transform::chi_r_to_chi_k(chi_r, lat);
                hdf5::write_complex_cube(group_id, dataset_name, chi_k);
            }
        }
        
        // Close the group
        H5Gclose(group_id);
    }
    
public:    
    void jacknifeAnalysis() {
        // No-op as requested
    }
};

#endif