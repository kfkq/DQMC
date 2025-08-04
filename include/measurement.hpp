#ifndef MEASUREMENT_HPP
#define MEASUREMENT_HPP

#include "dqmc.hpp"
#include "utility.hpp"

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

// Generalized data structure for both equal- and unequal-time observables.
struct DataRow {
    int tau;                  // Set to 0 for equal-time cases
    double coord1, coord2;    // (dx, dy) or (kx, ky)
    int a, b;
    double re_mean, re_error;
    double im_mean, im_error; // For real data, im_* fields are 0
};

namespace io {
    // --- Private Helper for Writing Text Files ---
    // A generic writer that takes a header-writing lambda and a row-writing lambda.
    template<typename HeaderFunc, typename RowFunc>
    static void write_stats_file(const std::string& filename,
                                 const std::vector<DataRow>& results,
                                 HeaderFunc write_header,
                                 RowFunc write_row) {
        std::ofstream out(filename);
        if (!out) {
            throw std::runtime_error("Could not open file for writing stats: " + filename);
        }
        out << std::fixed << std::setprecision(12);
        write_header(out); // Call the lambda to write the specific header
        for (const auto& res : results) {
            write_row(out, res); // Call the lambda to write the specific row format
        }
    }

    // Append a single DataRow to a binary file.
    inline void save_bin_data(const std::string& filename, const DataRow& row) {
        std::ofstream out(filename, std::ios::binary | std::ios::app);
        if (!out) {
            throw std::runtime_error("Could not open file for binary writing: " + filename);
        }
        out.write(reinterpret_cast<const char*>(&row), sizeof(DataRow));
    }

    // Overload to append a vector of DataRows to a binary file.
    inline void save_bin_data(const std::string& filename, const std::vector<DataRow>& data) {
        if (data.empty()) return;
        std::ofstream out(filename, std::ios::binary | std::ios::app);
        if (!out) {
            throw std::runtime_error("Could not open file for binary writing: " + filename);
        }
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(DataRow));
    }

    // Load an entire binary file of DataRows.
    inline std::vector<DataRow> load_bin_data(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            // This is not an error, the file might not exist yet.
            return {};
        }

        in.seekg(0, std::ios::end);
        const std::streampos fileSize = in.tellg();
        in.seekg(0, std::ios::beg);

        if (fileSize == 0) return {};

        const size_t num_records = fileSize / sizeof(DataRow);
        std::vector<DataRow> data(num_records);
        in.read(reinterpret_cast<char*>(data.data()), fileSize);
        return data;
    }

    // Save final scalar statistics to a text file.
    inline void save_scalar_stats(const std::string& filename, double mean, double error) {
        std::ofstream out(filename);
        if (!out) {
            throw std::runtime_error("Could not open file for writing stats: " + filename);
        }
        out << std::fixed << std::setprecision(12)
            << std::setw(20) << "mean"
            << std::setw(20) << "error\n";
        out << std::setw(20) << mean
            << std::setw(20) << error << '\n';
    }

    // Save final real-space statistics to a text file.
    inline void save_real_stats(const std::string& filename, const std::vector<DataRow>& results) {
        write_stats_file(filename, results,
            // Header-writing lambda
            [](std::ostream& os) {
                os << std::setw(20) << "dx" << std::setw(20) << "dy"
                   << std::setw(4)  << "a"  << std::setw(4)  << "b"
                   << std::setw(20) << "mean" << std::setw(20) << "error\n";
            },
            // Row-writing lambda
            [](std::ostream& os, const DataRow& res) {
                os << std::setw(20) << res.coord1 << std::setw(20) << res.coord2
                   << std::setw(4)  << res.a << std::setw(4)  << res.b
                   << std::setw(20) << res.re_mean << std::setw(20) << res.re_error << '\n';
            }
        );
    }

    // Save final k-space statistics to a text file.
    inline void save_complex_stats(const std::string& filename, const std::vector<DataRow>& results) {
        write_stats_file(filename, results,
            [](std::ostream& os) {
                os << std::setw(20) << "kx" << std::setw(20) << "ky"
                   << std::setw(4)  << "a"  << std::setw(4)  << "b"
                   << std::setw(20) << "re_mean" << std::setw(20) << "re_error"
                   << std::setw(20) << "im_mean" << std::setw(20) << "im_error\n";
            },
            [](std::ostream& os, const DataRow& res) {
                os << std::setw(20) << res.coord1 << std::setw(20) << res.coord2
                   << std::setw(4)  << res.a << std::setw(4)  << res.b
                   << std::setw(20) << res.re_mean << std::setw(20) << res.re_error
                   << std::setw(20) << res.im_mean << std::setw(20) << res.im_error << '\n';
            }
        );
    }

    // Save final real-space statistics for unequal-time data to a text file.
    inline void save_real_tau_stats(const std::string& filename, const std::vector<DataRow>& results) {
        write_stats_file(filename, results,
            [](std::ostream& os) {
                os << std::setw(8)  << "tau" << std::setw(20) << "dx" << std::setw(20) << "dy"
                   << std::setw(4)  << "a"   << std::setw(4)  << "b"
                   << std::setw(20) << "mean" << std::setw(20) << "error\n";
            },
            [](std::ostream& os, const DataRow& res) {
                os << std::setw(8)  << res.tau << std::setw(20) << res.coord1 << std::setw(20) << res.coord2
                   << std::setw(4)  << res.a << std::setw(4)  << res.b
                   << std::setw(20) << res.re_mean << std::setw(20) << res.re_error << '\n';
            }
        );
    }

    // Save final k-space statistics for unequal-time data to a text file.
    inline void save_complex_tau_stats(const std::string& filename, const std::vector<DataRow>& results) {
        write_stats_file(filename, results,
            [](std::ostream& os) {
                os << std::setw(8)  << "tau" << std::setw(20) << "kx" << std::setw(20) << "ky"
                   << std::setw(4)  << "a"   << std::setw(4)  << "b"
                   << std::setw(20) << "re_mean" << std::setw(20) << "re_error"
                   << std::setw(20) << "im_mean" << std::setw(20) << "im_error\n";
            },
            [](std::ostream& os, const DataRow& res) {
                os << std::setw(8) << res.tau << std::setw(20) << res.coord1 << std::setw(20) << res.coord2
                   << std::setw(4) << res.a << std::setw(4) << res.b
                   << std::setw(20) << res.re_mean << std::setw(20) << res.re_error
                   << std::setw(20) << res.im_mean << std::setw(20) << res.im_error << '\n';
            }
        );
    }
} // namespace io

namespace transform {
    inline int pbc_shortest(int d, int L) {                                                                                                                                                                                       
        if (d >  L/2)  d -= L;                                                                                                                                                                                         
        if (d <= -L/2) d += L;                                                                                                                                                                                         
        return d;                                                                                                                                                                                                      
    } 
    // ------------------------------------------------------------------           
    //  chi_site_to_chi_r                                                           
    //  Convert site–site correlator \chi(i,j) → \chi_{a,b}(dx,dy)                        
    //  dx,dy range: (-Lx/2+1 … Lx/2)  and  (-Ly/2+1 … Lx/2)                        
    // ------------------------------------------------------------------           
    inline
    std::vector<DataRow>
    chi_site_to_chi_r(const arma::cube& chi_site, const Lattice& lat) {
        const int n_orb = lat.n_orb();                         
        const int Lx = lat.Lx();                                                    
        const int Ly = lat.Ly();
        const int n_cells = lat.size();                                            
        const int n_tau = chi_site.n_slices;

        // Use a map to accumulate values for each unique (tau, dx, dy, a, b) tuple
        std::map<std::tuple<int, int, int, int, int>, double> accumulator;

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
                accumulator[{tau, dx_idx, dy_idx, a, b}] += val;
            }
        }

        // Convert the accumulated map to a vector of DataRow
        std::vector<DataRow> data_rows;
        data_rows.reserve(accumulator.size());
        const auto& a1 = lat.a1();
        const auto& a2 = lat.a2();
        for (const auto& [key, accumulated_val] : accumulator) {
            const auto [tau, dx_idx, dy_idx, a, b] = key;
            double dx_phys = (dx_idx - (Lx/2 - 1)) * a1[0] + (dy_idx - (Ly/2 - 1)) * a2[0];
            double dy_phys = (dx_idx - (Lx/2 - 1)) * a1[1] + (dy_idx - (Ly/2 - 1)) * a2[1];
            data_rows.emplace_back(DataRow{tau, dx_phys, dy_phys, a, b, accumulated_val / n_cells, 0.0, 0.0, 0.0});
        }
        return data_rows;
    }

    // Overload for equal-time case (arma::mat)
    inline std::vector<DataRow> chi_site_to_chi_r(const arma::mat& chi_site, const Lattice& lat) {
        // Wrap the matrix in a 1-slice cube and call the main implementation
        arma::cube temp_cube(chi_site.n_rows, chi_site.n_cols, 1);
        temp_cube.slice(0) = chi_site;
        return chi_site_to_chi_r(temp_cube, lat);
    }

    inline std::vector<DataRow> chi_r_to_chi_k(
        const std::vector<DataRow>& chi_r_rows,
        const Lattice& lat)
    {
        const auto& kpts = lat.k_points();
        const int nk = static_cast<int>(kpts.size());

        std::vector<DataRow> k_space_data;
        if (chi_r_rows.empty()) return k_space_data;

        // Group real-space data by (tau, a, b) to perform FT on each channel
        std::map<std::tuple<int, int, int>, std::vector<const DataRow*>> grouped_r_data;
        for (const auto& row : chi_r_rows) {
            grouped_r_data[{row.tau, row.a, row.b}].push_back(&row);
        }

        for (int kidx = 0; kidx < nk; ++kidx) {
            const auto& k = kpts[kidx];
            for (const auto& [key, rows] : grouped_r_data) {
                const auto [tau, a, b] = key;
                std::complex<double> chi_k_val(0.0, 0.0);
                for (const auto* r_row : rows) {
                    double x = r_row->coord1;
                    double y = r_row->coord2;
                    double phase = k[0]*x + k[1]*y;
                    chi_k_val += r_row->re_mean * std::complex<double>(std::cos(phase), -std::sin(phase));
                }
                k_space_data.emplace_back(DataRow{tau, k[0], k[1], a, b, chi_k_val.real(), 0.0, chi_k_val.imag(), 0.0});
            }
        }
        return k_space_data;
    }
}

namespace statistics {
    inline double mean(const std::vector<double>& v) {                                                                                                                                                             
        return v.empty() ? 0.0 :                                                                                                                                                                                   
               std::accumulate(v.begin(), v.end(), 0.0) / v.size();                                                                                                                                                
    }
    
    struct JackknifeResult {
        std::vector<double> means;
        std::vector<double> errors;
    };

    inline JackknifeResult jackknife(const std::vector<double>& data) {
        const std::size_t N = data.size();
        JackknifeResult res;
        res.means.resize(N);
        res.errors.resize(N);

        for (std::size_t k = 0; k < N; ++k) {
            double sum = 0.0, sum_sq = 0.0;
            for (std::size_t j = 0; j < N; ++j) {
                if (j == k) continue;
                sum    += data[j];
                sum_sq += data[j] * data[j];
            }
            const double m   = sum / (N - 1);
            const double var = (sum_sq / (N - 1)) - (m * m);
            res.means[k]  = m;
            res.errors[k] = std::sqrt(var / (N - 2));
        }
        return res;
    }
} // namespace statistics

class scalarObservable {
private:
    std::string filename_;

    // accumulation
    double local_sum_ = 0.0;
    double global_sum_ = 0.0;

    int local_count_ = 0;
    int global_count_ = 0;

public:
    scalarObservable(const std::string& filename, int rank)
        : filename_("results/" + filename) {
        if (!utility::ensure_dir("results", rank)) {
            throw std::runtime_error("Could not create results directory");
        }
    }

    const std::string& filename() const { return filename_; }
    
    void operator+=(double value) {
        local_sum_ += value;
        local_count_++;
    }
    
    void accumulate(MPI_Comm comm, int rank) {
        MPI_Allreduce(&local_sum_, &global_sum_, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_count_, &global_count_, 1, MPI_INT, MPI_SUM, comm);
        
        if (rank == 0) {
            DataRow row;
            row.re_mean = (global_count_ > 0) ? (global_sum_ / global_count_) : 0.0;
            io::save_bin_data(filename_ + ".bins", row);
        }

        // reset accumulators
        local_sum_ = 0.0;
        local_count_ = 0;
    }

    void jackknife(int rank) {
        if (rank != 0) return;

        auto binned_data = io::load_bin_data(filename_ + ".bins");
        if (binned_data.empty()) {
            std::cerr << "Warning: no data found in " << filename_ << ".bins for jackknife\n";
            return;
        }

        std::vector<double> values;
        values.reserve(binned_data.size());
        for (const auto& row : binned_data) {
            values.push_back(row.re_mean);
        }

        const auto [means, errors] = statistics::jackknife(values);                                                                                                                                                
        const double mMean  = statistics::mean(means);
        const double mError = statistics::mean(errors);
        io::save_scalar_stats(filename_ + ".stat", mMean, mError);
    }
    
};

class equalTimeObservable {
private:
    std::string filename_;
    int matrix_size_ = 0;  // inferred on first matrix

    arma::mat local_sum_;
    int local_count_ = 0;

public:
    equalTimeObservable(const std::string& filename, int rank) {
        filename_ = "results/" + filename;
    }

    const std::string& filename() const { return filename_; }

    void operator+=(const arma::mat& m) {
        if (matrix_size_ == 0) {
            matrix_size_ = m.n_rows;
            local_sum_.set_size(matrix_size_, matrix_size_);
            local_sum_.zeros();
        }
        local_sum_ += m;
        ++local_count_;
    }

    void accumulate(const Lattice& lat, MPI_Comm comm, int rank) {        
        const int n_sites = local_sum_.n_rows;
        arma::mat global_sum(n_sites, n_sites, arma::fill::zeros);
        int global_count = 0;


        MPI_Allreduce(local_sum_.memptr(), global_sum.memptr(),
                      n_sites * n_sites, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_count_, &global_count, 1, MPI_INT, MPI_SUM, comm);

        if (rank == 0) {
            arma::mat chi_site = global_sum / global_count;
            auto data = transform::chi_site_to_chi_r(chi_site, lat);
            io::save_bin_data(filename_ + "_R.bins", data);
        }

        local_sum_.zeros();
        local_count_ = 0;
    }

    void jackknife(int rank)
    {
        if (rank != 0) return;
        /* --------------------------------------------------------------- */
        /*  1. Real-space correlator χ(r)                                  */
        /* --------------------------------------------------------------- */
        auto binned_data_r = io::load_bin_data(filename_ + "_R.bins");
        if (binned_data_r.empty()) {
            std::cerr << "Warning: no real-space data found for " << filename_ << "\n";
            return; 
        }

        // Group data by (dx, dy, a, b) key to prepare for jackknife
        std::map<std::tuple<double,double,int,int>, std::vector<double>> dataR;
        for (const auto& row : binned_data_r) {
            // The tau component (row.tau) is always 0 here, so we ignore it in the key
            dataR[{row.coord1, row.coord2, row.a, row.b}].push_back(row.re_mean);
        }
        
        // Process real-space data and write statistics
        if (!dataR.empty()) {
            // 2. Process data: perform jackknife and store results
            std::vector<DataRow> resultsR;
            resultsR.reserve(dataR.size());

            for (const auto& [key, vals] : dataR) {
                if (vals.size() < 2) continue;
                const auto [means, errs] = statistics::jackknife(vals);
                const double mean_val  = statistics::mean(means);
                const double error_val = statistics::mean(errs);
                const auto [dx,dy,a,b] = key;
                resultsR.emplace_back(DataRow{0, dx, dy, a, b, mean_val, error_val, 0.0, 0.0});
            }

            io::save_real_stats(filename_ + ".statR", resultsR);
        }                                                                                                                                                                                                          
                                                                                                                                                                                                                   
        /* --------------------------------------------------------------- */                                                                                                                                      
        /*  2. k-space correlator χ(k)                                     */                                                                                                                                      
        /* --------------------------------------------------------------- */                                                                                                                                      
        auto binned_data_k = io::load_bin_data(filename_ + "_K.bins");
        if (binned_data_k.empty()) {
            // This is normal if fourierTransform hasn't been run
            return;
        }

        // Group data by (kx, ky, a, b) key
        std::map<std::tuple<double,double,int,int>,                                                                                                                                                                
                 std::pair<std::vector<double>,std::vector<double>>> dataK;                                                                                                                                        
        for (const auto& row : binned_data_k) {
            auto key = std::make_tuple(row.coord1, row.coord2, row.a, row.b);
            dataK[key].first.push_back(row.re_mean);
            dataK[key].second.push_back(row.im_mean);
        }                                                                                                                                                                                                          
                                                                                                                                                                                                                   
        // Process k-space data and write statistics
        if (!dataK.empty()) {
            // 2. Process data: perform jackknife and store results
            std::vector<DataRow> resultsK;
            resultsK.reserve(dataK.size());

            for (const auto& [key, vecs] : dataK) {
                const auto& [re_vals, im_vals] = vecs;
                if (re_vals.size() < 2) continue;

                const auto [re_means, re_errs] = statistics::jackknife(re_vals);
                const auto [im_means, im_errs] = statistics::jackknife(im_vals);

                const double reMean  = statistics::mean(re_means);
                const double reError = statistics::mean(re_errs);
                const double imMean  = statistics::mean(im_means);
                const double imError = statistics::mean(im_errs);

                const auto [kx,ky,a,b] = key;
                resultsK.emplace_back(DataRow{0, kx, ky, a, b, reMean, reError, imMean, imError});
            }

            io::save_complex_stats(filename_ + ".statK", resultsK);
        }                                                                                                                                                                                                          
    } 

    // ------------------------------------------------------------------
    //  χ(r) → χ(k)  Fourier transform for every stored bin
    // ------------------------------------------------------------------
    void fourierTransform(const Lattice& lat, int rank)
    {
        if (rank != 0) return;

        // Load all real-space bins at once
        auto binned_data_r = io::load_bin_data(filename_ + "_R.bins");
        if (binned_data_r.empty()) return;

        // Group data by bin. Since we don't have a bin index, we need to infer it.
        // We can do this by finding the number of unique (dx, dy, a, b) points.
        std::set<std::tuple<double, double, int, int>> unique_points;
        for (const auto& row : binned_data_r) {
            unique_points.insert({row.coord1, row.coord2, row.a, row.b});
        }
        const size_t points_per_bin = unique_points.size();
        if (points_per_bin == 0) return;
        const size_t num_bins = binned_data_r.size() / points_per_bin;

        std::vector<DataRow> all_k_space_data;
        all_k_space_data.reserve(binned_data_r.size());

        // Process one bin at a time
        for (size_t i = 0; i < num_bins; ++i) {
            // Read bin rows into a contiguous vector
            std::vector<DataRow> chi_r_rows;
            chi_r_rows.reserve(points_per_bin);
            for (size_t j = 0; j < points_per_bin; ++j) {
                chi_r_rows.push_back(binned_data_r[i * points_per_bin + j]);
            }

            // Perform the Fourier Transform for this bin using DataRow pipeline
            auto k_space_bin_data = transform::chi_r_to_chi_k(chi_r_rows, lat);
            all_k_space_data.insert(all_k_space_data.end(), k_space_bin_data.begin(), k_space_bin_data.end());
        }

        // Write all k-space data to a single binary file
        io::save_bin_data(filename_ + "_K.bins", all_k_space_data);
    }
};

class unequalTimeObservable {
private:
    std::string filename_;
    int matrix_size_ = 0;
    int n_tau_ = 0;

    arma::cube local_sum_;  // (rows, cols, slices) -> (matrix_size, matrix_size, n_tau)
    int local_count_ = 0;

public:
    unequalTimeObservable(const std::string& filename, int rank) {
        filename_ = "results/" + filename;
    }

    const std::string& filename() const { return filename_; }

    void operator+=(const std::vector<arma::mat>& tau_matrices) {
        if (matrix_size_ == 0) {
            matrix_size_ = tau_matrices[0].n_rows;
            n_tau_ = tau_matrices.size();
            local_sum_.zeros(matrix_size_, matrix_size_, n_tau_);
        }
        
        for (int tau = 0; tau < n_tau_; ++tau) {
            local_sum_.slice(tau) += tau_matrices[tau];
        }
        ++local_count_;
    }

    void accumulate(const Lattice& lat, MPI_Comm comm, int rank) {
        const int n_sites = local_sum_.n_rows;
        arma::cube global_sum(n_sites, n_sites, n_tau_, arma::fill::zeros);
        int global_count = 0;

        // A single MPI call for the entire cube's data
        const int total_elements = n_sites * n_sites * n_tau_;
        MPI_Allreduce(local_sum_.memptr(), global_sum.memptr(),
                      total_elements, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_count_, &global_count, 1, MPI_INT, MPI_SUM, comm);

        if (rank == 0) {
            // Force evaluation of the expression to resolve overload ambiguity
            arma::cube chi_site = global_sum / global_count;
            auto data = transform::chi_site_to_chi_r(chi_site, lat);
            io::save_bin_data(filename_ + "_R.bins", data);
        }

        // Reset accumulators
        local_sum_.zeros();
        local_count_ = 0;
    }

    void fourierTransform(const Lattice& lat, int rank) {
        if (rank != 0) return;

        auto binned_data_r = io::load_bin_data(filename_ + "_R.bins");
        if (binned_data_r.empty()) return;

        // Infer number of bins by finding number of unique points per bin
        std::set<std::tuple<int, double, double, int, int>> unique_points;
        for (const auto& row : binned_data_r) {
            unique_points.insert({row.tau, row.coord1, row.coord2, row.a, row.b});
        }
        const size_t points_per_bin = unique_points.size();
        if (points_per_bin == 0) return;
        const size_t num_bins = binned_data_r.size() / points_per_bin;

        std::vector<DataRow> all_k_space_data;
        all_k_space_data.reserve(binned_data_r.size());

        // Process one bin at a time
        for (size_t i = 0; i < num_bins; ++i) {
            // Reconstruct the chi_r field for this bin grouped by tau using DataRow
            std::map<int, std::vector<DataRow>> chi_tau_rows;

            for (size_t j = 0; j < points_per_bin; ++j) {
                const auto& row = binned_data_r[i * points_per_bin + j];
                chi_tau_rows[row.tau].push_back(row);
            }

            // Perform the Fourier Transform for this bin for each tau
            for (const auto& [tau, rows] : chi_tau_rows) {
                auto k_space_bin_data = transform::chi_r_to_chi_k(rows, lat);
                // ensure tau is propagated (already inside rows)
                all_k_space_data.insert(all_k_space_data.end(), k_space_bin_data.begin(), k_space_bin_data.end());
            }
        }

        io::save_bin_data(filename_ + "_K.bins", all_k_space_data);
    }

    void jackknife(int rank) {
        if (rank != 0) return;

        // --- Part 1: Real-space Jackknife ---
        auto binned_data_r = io::load_bin_data(filename_ + "_R.bins");
        if (binned_data_r.empty()) {
            std::cerr << "Warning: no real-space data found for " << filename_ << "\n";
            return;
        }

        std::map<std::tuple<int,double,double,int,int>, std::vector<double>> dataR;
        std::map<int, std::vector<double>> dataR0; // tau -> G_R0 values

        // Infer number of bins to correctly sum R0 data
        std::set<std::tuple<int, double, double, int, int>> unique_points;
        for (const auto& row : binned_data_r) {
            unique_points.insert({row.tau, row.coord1, row.coord2, row.a, row.b});
        }
        const size_t points_per_bin = unique_points.size();
        if (points_per_bin == 0) return;
        const size_t num_bins = binned_data_r.size() / points_per_bin;

        dataR0.clear();
        for (size_t i = 0; i < num_bins; ++i) {
            std::map<int, double> binR0_sum;
            for (size_t j = 0; j < points_per_bin; ++j) {
                const auto& row = binned_data_r[i * points_per_bin + j];
                dataR[{row.tau, row.coord1, row.coord2, row.a, row.b}].push_back(row.re_mean);
                if (std::abs(row.coord1) < 1e-12 && std::abs(row.coord2) < 1e-12) {
                    binR0_sum[row.tau] += row.re_mean;
                }
            }
            for (const auto& [tau, sum] : binR0_sum) {
                dataR0[tau].push_back(sum);
            }
        }
        if (!dataR.empty()) {
            // Calculate statistics
            std::vector<DataRow> resultsR;
            resultsR.reserve(dataR.size());
            for (const auto& [key, vals] : dataR) {
                if (vals.size() < 2) continue;
                const auto [means, errs] = statistics::jackknife(vals);
                const double mMean = statistics::mean(means);
                const double mError = statistics::mean(errs);
                const auto [tau,dx,dy,a,b] = key;
                resultsR.emplace_back(DataRow{tau, dx, dy, a, b, mMean, mError, 0.0, 0.0});
            }

            io::save_real_tau_stats(filename_ + ".statR", resultsR);
        }

        // --- Part 2: R0 Jackknife (Sum over orbitals at r=0) ---
        if (!dataR0.empty()) {
            // Calculate statistics
            struct StatResultR0 { int tau; double mean, error; };
            std::vector<StatResultR0> resultsR0;
            for (const auto& [tau, vals] : dataR0) {
                if (vals.size() < 2) continue;
                const auto [means, errs] = statistics::jackknife(vals);
                resultsR0.emplace_back(StatResultR0{tau, statistics::mean(means), statistics::mean(errs)});
            }

            std::string outnameR0 = filename_ + ".statR0";
            std::ofstream outR0(outnameR0);
            outR0 << std::fixed << std::setprecision(12)
                  << std::setw(8)  << "tau"
                  << std::setw(20) << "mean"
                  << std::setw(20) << "error\n";

            for (const auto& res : resultsR0) {
                outR0 << std::setw(8)  << res.tau
                      << std::setw(20) << res.mean
                      << std::setw(20) << res.error << '\n';
            }
        }

        // --- Part 3: K-space Jackknife ---
        auto binned_data_k = io::load_bin_data(filename_ + "_K.bins");
        if (binned_data_k.empty()) return;

        std::map<std::tuple<int,double,double,int,int>,
                 std::pair<std::vector<double>,std::vector<double>>> dataK;
        for (const auto& row : binned_data_k) {
            auto key = std::make_tuple(row.tau, row.coord1, row.coord2, row.a, row.b);
            dataK[key].first.push_back(row.re_mean);
            dataK[key].second.push_back(row.im_mean);
        }

        if (!dataK.empty()) {
            // Calculate statistics
            std::vector<DataRow> resultsK;
            resultsK.reserve(dataK.size());
            for (const auto& [key, vecs] : dataK) {
                const auto& [re_vals, im_vals] = vecs;
                if (re_vals.size() < 2) continue;

                const auto [re_means, re_errs] = statistics::jackknife(re_vals);
                const auto [im_means, im_errs] = statistics::jackknife(im_vals);

                const double reMean = statistics::mean(re_means);
                const double reError = statistics::mean(re_errs);
                const double imMean = statistics::mean(im_means);
                const double imError = statistics::mean(im_errs);

                const auto [tau,kx,ky,a,b] = key;
                resultsK.emplace_back(DataRow{tau, kx, ky, a, b, reMean, reError, imMean, imError});
            }

            io::save_complex_tau_stats(filename_ + ".statK", resultsK);
        }
    }
};

class MeasurementManager {
private:                                                                                                                                                                                                           
    std::vector<scalarObservable> scalarObservables_;                                                                                                                                                              
    std::vector<std::function<double(const std::vector<GF>&, const Lattice&)>> scalarCalculators_;                                                                                                                       
    MPI_Comm comm_;                                                                                                                                                                                                
    int rank_;                                                                                                                                                                                                     
                                                                                                                                                                                                                   
    std::vector<equalTimeObservable> equalTimeObservables_;                                                                                                                                                        
    std::vector<std::function<arma::mat(const std::vector<GF>&, const Lattice&)>> eqTimeCalculators_; 

    std::vector<unequalTimeObservable> unequalTimeObservables_;
    std::vector<std::function<std::vector<arma::mat>(const std::vector<GF>&, const Lattice&)>> uneqTimeCalculators_;
    
public:
    MeasurementManager(MPI_Comm comm, int rank) : comm_(comm), rank_(rank) {}

    void addScalar(const std::string& name, 
             std::function<double(const std::vector<GF>&, const Lattice& lat)> calculator) {
        scalarObservables_.emplace_back(name, rank_);
        scalarCalculators_.push_back(calculator);
    }

    void addEqualTime(const std::string& name,
                      std::function<arma::mat(const std::vector<GF>&, const Lattice& lat)> calculator) {
        equalTimeObservables_.emplace_back(name, rank_);
        eqTimeCalculators_.push_back(calculator);
    }

    void addUnequalTime(const std::string& name,
                        std::function<std::vector<arma::mat>(const std::vector<GF>&, const Lattice&)> calculator) {
        unequalTimeObservables_.emplace_back(name, rank_);
        uneqTimeCalculators_.push_back(calculator);
    }

    void measure(const std::vector<GF>& greens, const Lattice& lat) {
        for (size_t i = 0; i < scalarCalculators_.size(); ++i) {
            scalarObservables_[i] += scalarCalculators_[i](greens, lat);
        }
        for (size_t i = 0; i < eqTimeCalculators_.size(); ++i) {
            equalTimeObservables_[i] += eqTimeCalculators_[i](greens, lat);
        }
    }

    void measure_unequalTime(const std::vector<GF>& greens, const Lattice& lat) {
        for (size_t i = 0; i < uneqTimeCalculators_.size(); ++i) {
            unequalTimeObservables_[i] += uneqTimeCalculators_[i](greens, lat);
        }
    }

    void accumulate(const Lattice& lat) {
        for (auto& obs : scalarObservables_) {
            obs.accumulate(comm_, rank_);
            
        }
        for (auto& obs : equalTimeObservables_) {
            obs.accumulate(lat, comm_, rank_);
        }
        for (auto& obs : unequalTimeObservables_) {
            obs.accumulate(lat, comm_, rank_);
        }
    }

    // Fourier transform
    void fourierTransform(const Lattice& lat) {
        for (auto& obs : equalTimeObservables_) {
            obs.fourierTransform(lat, rank_);
        }
        for (auto& obs : unequalTimeObservables_) {
            obs.fourierTransform(lat, rank_);
        }
        MPI_Barrier(comm_);
    }

    // jackknife analysis
    void jacknifeAnalysis() {
        for (auto& obs : scalarObservables_) {
            obs.jackknife(rank_);
        }
        for (auto& obs : equalTimeObservables_) {
            obs.jackknife(rank_);
        }
        for (auto& obs : unequalTimeObservables_) {
            obs.jackknife(rank_);
        }
        MPI_Barrier(comm_);
    }
};

#endif
