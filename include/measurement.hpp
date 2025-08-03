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
    // Append a single DataRow to a binary file.
    inline void save_bin_data(const std::string& filename, const DataRow& row) {
        std::ofstream out(filename, std::ios::binary | std::ios::app);
        if (!out) {
            throw std::runtime_error("Could not open file for binary writing: " + filename);
        }
        out.write(reinterpret_cast<const char*>(&row), sizeof(DataRow));
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
        const arma::field<arma::mat>& chi_r,
        const Lattice& lat,
        int tau = 0)
    {
        const auto& kpts = lat.k_points();
        const int nk = static_cast<int>(kpts.size());
        const int n_orb = lat.n_orb();
        const int Lx = lat.Lx();
        const int Ly = lat.Ly();

        std::vector<DataRow> k_space_data;
        k_space_data.reserve(nk * n_orb * n_orb);

        const std::array<double,2>& a1 = lat.a1();
        const std::array<double,2>& a2 = lat.a2();
        const double invN = 1.0 / (Lx * Ly);

        for (int kidx = 0; kidx < nk; ++kidx) {
            const auto& k = kpts[kidx];
            for (int a = 0; a < n_orb; ++a) {
                for (int b = 0; b < n_orb; ++b) {
                    std::complex<double> chi_k_val(0.0, 0.0);
                    for (int rx = 0; rx < Lx; ++rx) {
                        for (int ry = 0; ry < Ly; ++ry) {
                            double x = (rx - Lx/2 + 1) * a1[0] + (ry - Ly/2 + 1) * a2[0];
                            double y = (rx - Lx/2 + 1) * a1[1] + (ry - Ly/2 + 1) * a2[1];
                            double phase = k[0]*x + k[1]*y;
                            chi_k_val += chi_r(a,b)(rx,ry) * std::complex<double>(std::cos(phase), -std::sin(phase));
                        }
                    }
                    chi_k_val *= invN;
                    k_space_data.emplace_back(DataRow{tau, k[0], k[1], a, b, chi_k_val.real(), 0.0, chi_k_val.imag(), 0.0});
                }
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
    equalTimeObservable(const std::string& filename, int rank)
        : filename_(filename){
        if (!utility::ensure_dir("results/" + filename_, rank)) {
            throw std::runtime_error("Could not create results/" + filename_ + " directory");
        }
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

        static int bin_counter = 0;
        if (rank == 0) {
            // --- 1. Data Collection ---
            // Force evaluation of the expression to resolve overload ambiguity
            arma::mat chi_site = global_sum / global_count;
            // Now the call is unambiguous
            auto data = transform::chi_site_to_chi_r(chi_site, lat);

            // --- 2. File Writing ---
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "results/%s/binR_%04d.dat", filename_.c_str(), bin_counter++);

            std::ofstream out(fname);
            out << std::fixed << std::setprecision(12)
                << std::setw(20) << "dx"
                << std::setw(20) << "dy"
                << std::setw(4)  << "a"
                << std::setw(4)  << "b"
                << std::setw(20) << "mean" << '\n';

            for (const auto& row : data) {
                out << std::setw(20) << row.coord1
                    << std::setw(20) << row.coord2
                    << std::setw(4)  << row.a
                    << std::setw(4)  << row.b
                    << std::setw(20) << std::setprecision(12)
                    << row.re_mean << '\n';
            }
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
        std::vector<std::string> binsR;
        for (int idx = 0; ; ++idx) {
            char fname[256];
            std::snprintf(fname, sizeof(fname), "results/%s/binR_%04d.dat", filename_.c_str(), idx);
            if (!std::ifstream(fname).good()) break;
            binsR.emplace_back(fname);                                                                                                                                                                             
        }
        
        std::map<std::tuple<double,double,int,int>, std::vector<double>> dataR;
        for (const std::string& bin : binsR) {
            std::ifstream in(bin);
            std::string line; std::getline(in, line);
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                double dx, dy; int a, b; double v;
                if (iss >> dx >> dy >> a >> b >> v) {
                    dataR[{dx,dy,a,b}].push_back(v);
                }
            }
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

            // 3. Write results to file
            char outnameR[256];
            std::snprintf(outnameR, sizeof(outnameR),
                          "results/%s/statR.dat", filename_.c_str());
            std::ofstream outR(outnameR);
            outR << std::fixed << std::setprecision(12)
                 << std::setw(20) << "dx"
                 << std::setw(20) << "dy"
                 << std::setw(4)  << "a"
                 << std::setw(4)  << "b"
                 << std::setw(20) << "mean"
                 << std::setw(20) << "error\n";

            for (const auto& res : resultsR) {
                outR << std::setw(20) << res.coord1
                     << std::setw(20) << res.coord2
                     << std::setw(4)  << res.a
                     << std::setw(4)  << res.b
                     << std::setw(20) << res.re_mean
                     << std::setw(20) << res.re_error << '\n';
            }
        }                                                                                                                                                                                                          
                                                                                                                                                                                                                   
        /* --------------------------------------------------------------- */                                                                                                                                      
        /*  2. k-space correlator χ(k)                                     */                                                                                                                                      
        /* --------------------------------------------------------------- */                                                                                                                                      
        std::vector<std::string> binsK;                                                                                                                                                                            
        for (int idx = 0; ; ++idx) {                                                                                                                                                                               
            char fname[256];                                                                                                                                                                                       
            std::snprintf(fname, sizeof(fname),                                                                                                                                                                    
                          "results/%s/binK_%04d.dat", filename_.c_str(), idx);                                                                                                                                     
            if (!std::ifstream(fname).good()) break;                                                                                                                                                               
            binsK.emplace_back(fname);                                                                                                                                                                             
        }                                                                                                                                                                                                          
                                                                                                                                                                                                                   
        std::map<std::tuple<double,double,int,int>,                                                                                                                                                                
                 std::pair<std::vector<double>,std::vector<double>>> dataK;                                                                                                                                        
        for (const std::string& bin : binsK) {                                                                                                                                                                     
            std::ifstream in(bin);                                                                                                                                                                                 
            std::string line; std::getline(in, line);          // header                                                                                                                                           
            while (std::getline(in, line)) {                                                                                                                                                                       
                if (line.empty() || line[0] == '#') continue;                                                                                                                                                      
                std::istringstream iss(line);                                                                                                                                                                      
                double kx, ky; int a, b; double re, im;                                                                                                                                                            
                if (iss >> kx >> ky >> a >> b >> re >> im) {                                                                                                                                                       
                    dataK[{kx,ky,a,b}].first .push_back(re);                                                                                                                                                       
                    dataK[{kx,ky,a,b}].second.push_back(im);                                                                                                                                                       
                }                                                                                                                                                                                                  
            }                                                                                                                                                                                                      
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

            // 3. Write results to file
            char outnameK[256];
            std::snprintf(outnameK, sizeof(outnameK),
                          "results/%s/statK.dat", filename_.c_str());
            std::ofstream outK(outnameK);
            outK << std::fixed << std::setprecision(12)
                 << std::setw(20) << "kx"
                 << std::setw(20) << "ky"
                 << std::setw(4)  << "a"
                 << std::setw(4)  << "b"
                 << std::setw(20) << "re_mean"
                 << std::setw(20) << "re_error"
                 << std::setw(20) << "im_mean"
                 << std::setw(20) << "im_error\n";

            for (const auto& res : resultsK) {
                outK << std::setw(20) << res.coord1
                     << std::setw(20) << res.coord2
                     << std::setw(4)  << res.a
                     << std::setw(4)  << res.b
                     << std::setw(20) << res.re_mean
                     << std::setw(20) << res.re_error
                     << std::setw(20) << res.im_mean
                     << std::setw(20) << res.im_error << '\n';
            }
        }                                                                                                                                                                                                          
    } 

    // ------------------------------------------------------------------
    //  χ(r) → χ(k)  Fourier transform for every stored bin
    // ------------------------------------------------------------------
    void fourierTransform(const Lattice& lat, int rank)
    {
        if (rank != 0) return;

        const auto& kpts = lat.k_points();        // already shifted to (-π,π]
        const int  nk    = static_cast<int>(kpts.size());
        const int  n_orb = lat.n_orb();

        int bin_idx = 0;
        while (true) {
            char rname[256];
            std::snprintf(rname, sizeof(rname),
                          "results/%s/binR_%04d.dat", filename_.c_str(), bin_idx);
            std::ifstream rin(rname);
            if (!rin.good()) break;                 // no more bins

            // read χ(r) into a cube (a,b,rx,ry)
            arma::field<arma::mat> chi_r(n_orb, n_orb);
            for (int a = 0; a < n_orb; ++a)
                for (int b = 0; b < n_orb; ++b)
                    chi_r(a,b).zeros(lat.Lx(), lat.Ly());

            std::string line;
            std::getline(rin, line);                // skip header
            while (std::getline(rin, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                double rx, ry; int a, b; double v;
                if (!(iss >> rx >> ry >> a >> b >> v)) continue;

                // map (rx,ry) back to array indices
                int ix = static_cast<int>(rx / lat.a1()[0] + lat.Lx()/2 - 1);
                int iy = static_cast<int>(ry / lat.a2()[1] + lat.Ly()/2 - 1);
                if (ix < 0 || ix >= lat.Lx()) continue;
                if (iy < 0 || iy >= lat.Ly()) continue;
                chi_r(a,b)(ix, iy) = v;
            }
            rin.close();

            // --- 2. Perform Fourier Transform by calling the shared function ---
            auto k_space_data = transform::chi_r_to_chi_k(chi_r, lat);

            // --- 3. Write K-Space Bin File ---
            char kname[256];
            std::snprintf(kname, sizeof(kname),
                          "results/%s/binK_%04d.dat", filename_.c_str(), bin_idx);
            std::ofstream kout(kname);
            kout << std::fixed << std::setprecision(12)
                 << std::setw(20) << "kx"
                 << std::setw(20) << "ky"
                 << std::setw(4)  << "a"
                 << std::setw(4)  << "b"
                 << std::setw(20) << "re_mean"
                 << std::setw(20) << "im_mean\n";

            for (const auto& row : k_space_data) {
                kout << std::setw(20) << row.coord1
                     << std::setw(20) << row.coord2
                     << std::setw(4)  << row.a
                     << std::setw(4)  << row.b
                     << std::setw(20) << row.re_mean
                     << std::setw(20) << row.im_mean << '\n';
            }
            ++bin_idx;
        }
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
    unequalTimeObservable(const std::string& filename, int rank)
        : filename_(filename) {
        if (!utility::ensure_dir("results/" + filename_, rank)) {
            throw std::runtime_error("Could not create results/" + filename_ + " directory");
        }
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

        static int bin_counter = 0;
        if (rank == 0) {
            // --- 1. Data Collection ---
            // Force evaluation of the expression to resolve overload ambiguity
            arma::cube chi_site = global_sum / global_count;
            // Now the call is unambiguous
            auto data = transform::chi_site_to_chi_r(chi_site, lat);

            // --- 2. File Writing ---
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "results/%s/binR_%04d.dat", filename_.c_str(), bin_counter++);
            std::ofstream out(fname);
            out << std::fixed << std::setprecision(12)
                << std::setw(8)  << "tau"
                << std::setw(20) << "dx"
                << std::setw(20) << "dy"
                << std::setw(4)  << "a"
                << std::setw(4)  << "b"
                << std::setw(20) << "value" << '\n';

            for (const auto& row : data) {
                out << std::setw(8)  << row.tau
                    << std::setw(20) << row.coord1
                    << std::setw(20) << row.coord2
                    << std::setw(4)  << row.a
                    << std::setw(4)  << row.b
                    << std::setw(20) << row.re_mean << '\n';
            }
        }

        // Reset accumulators
        local_sum_.zeros();
        local_count_ = 0;
    }

    void fourierTransform(const Lattice& lat, int rank) {
        if (rank != 0) return;

        const auto& kpts = lat.k_points();
        const int nk = static_cast<int>(kpts.size());
        const int n_orb = lat.n_orb();

        int bin_idx = 0;
        while (true) {
            char rname[256];
            std::snprintf(rname, sizeof(rname),
                          "results/%s/binR_%04d.dat", filename_.c_str(), bin_idx);
            std::ifstream rin(rname);
            if (!rin.good()) break;

            // Read χ(τ,r) into a field indexed by tau
            std::map<int, arma::field<arma::mat>> chi_tau_r;
            
            std::string line;
            std::getline(rin, line); // skip header
            while (std::getline(rin, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                int tau; double rx, ry; int a, b; double v;
                if (!(iss >> tau >> rx >> ry >> a >> b >> v)) continue;

                // Initialize field for this tau if needed
                if (chi_tau_r.find(tau) == chi_tau_r.end()) {
                    chi_tau_r[tau] = arma::field<arma::mat>(n_orb, n_orb);
                    for (int aa = 0; aa < n_orb; ++aa)
                        for (int bb = 0; bb < n_orb; ++bb)
                            chi_tau_r[tau](aa,bb).zeros(lat.Lx(), lat.Ly());
                }

                // Map (rx,ry) back to array indices
                int ix = static_cast<int>(rx / lat.a1()[0] + lat.Lx()/2 - 1);
                int iy = static_cast<int>(ry / lat.a2()[1] + lat.Ly()/2 - 1);
                if (ix < 0 || ix >= lat.Lx()) continue;
                if (iy < 0 || iy >= lat.Ly()) continue;
                chi_tau_r[tau](a,b)(ix, iy) = v;
            }
            rin.close();

            // --- 2. Perform Fourier Transform and Collect Data ---
            std::vector<DataRow> k_space_data;
            const std::array<double,2>& a1 = lat.a1();
            const std::array<double,2>& a2 = lat.a2();
            const double invN = 1.0 / (lat.Lx() * lat.Ly());

            for (const auto& [tau, chi_r] : chi_tau_r) {
                // Call the shared function for each tau slice
                auto single_tau_k_data = transform::chi_r_to_chi_k(chi_r, lat, tau);
                // Append the results to the main data vector
                k_space_data.insert(k_space_data.end(), single_tau_k_data.begin(), single_tau_k_data.end());
            }

            // --- 3. Write K-Space Bin File ---
            char kname[256];
            std::snprintf(kname, sizeof(kname),
                          "results/%s/binK_%04d.dat", filename_.c_str(), bin_idx);
            std::ofstream kout(kname);
            kout << std::fixed << std::setprecision(12)
                 << std::setw(8)  << "tau"
                 << std::setw(20) << "kx"
                 << std::setw(20) << "ky"
                 << std::setw(4)  << "a"
                 << std::setw(4)  << "b"
                 << std::setw(20) << "re_mean"
                 << std::setw(20) << "im_mean\n";

            for (const auto& row : k_space_data) {
                kout << std::setw(8)  << row.tau
                     << std::setw(20) << row.coord1
                     << std::setw(20) << row.coord2
                     << std::setw(4)  << row.a
                     << std::setw(4)  << row.b
                     << std::setw(20) << row.re_mean
                     << std::setw(20) << row.im_mean << '\n';
            }
            ++bin_idx;
        }
    }

    void jackknife(int rank) {
        if (rank != 0) return;

        // Real-space jackknife analysis
        std::vector<std::string> binsR;
        for (int idx = 0; ; ++idx) {
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "results/%s/binR_%04d.dat", filename_.c_str(), idx);
            if (!std::ifstream(fname).good()) break;
            binsR.emplace_back(fname);
        }

        std::map<std::tuple<int,double,double,int,int>, std::vector<double>> dataR;
        std::map<int, std::vector<double>> dataR0; // tau -> G_R0 values
        
        // No longer needed since we only use (dx=0, dy=0) values
        
        for (const std::string& bin : binsR) {
            std::ifstream in(bin);
            std::string line; std::getline(in, line); // header
            
            // Temporary storage for R0 data per bin (only dx=0, dy=0)
            std::map<int, double> binR0;
            
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                int tau; double dx, dy; int a, b; double v;
                if (iss >> tau >> dx >> dy >> a >> b >> v) {
                    dataR[{tau,dx,dy,a,b}].push_back(v);
                    
                    // Only accumulate for R0 when dx=0 and dy=0
                    if (std::abs(dx) < 1e-12 && std::abs(dy) < 1e-12) {
                        binR0[tau] += v;
                    }
                }
            }
            
            // Compute R0 average for this bin (sum over all orbitals at (0,0))
            for (const auto& [tau, sum] : binR0) {
                dataR0[tau].push_back(sum);
            }
        }

        // --- Part 1: Real-space Jackknife ---
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

            // Write to file
            char outnameR[256];
            std::snprintf(outnameR, sizeof(outnameR),
                          "results/%s/statR.dat", filename_.c_str());
            std::ofstream outR(outnameR);
            outR << std::fixed << std::setprecision(12)
                 << std::setw(8)  << "tau"
                 << std::setw(20) << "dx"
                 << std::setw(20) << "dy"
                 << std::setw(4)  << "a"
                 << std::setw(4)  << "b"
                 << std::setw(20) << "mean"
                 << std::setw(20) << "error\n";

            for (const auto& res : resultsR) {
                outR << std::setw(8)  << res.tau
                     << std::setw(20) << res.coord1
                     << std::setw(20) << res.coord2
                     << std::setw(4)  << res.a
                     << std::setw(4)  << res.b
                     << std::setw(20) << res.re_mean
                     << std::setw(20) << res.re_error << '\n';
            }
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

            // Write to file
            char outnameR0[256];
            std::snprintf(outnameR0, sizeof(outnameR0),
                          "results/%s/statR0.dat", filename_.c_str());
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
        std::vector<std::string> binsK;
        for (int idx = 0; ; ++idx) {
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "results/%s/binK_%04d.dat", filename_.c_str(), idx);
            if (!std::ifstream(fname).good()) break;
            binsK.emplace_back(fname);
        }

        std::map<std::tuple<int,double,double,int,int>,
                 std::pair<std::vector<double>,std::vector<double>>> dataK;
        for (const std::string& bin : binsK) {
            std::ifstream in(bin);
            std::string line; std::getline(in, line); // header
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                int tau; double kx, ky; int a, b; double re, im;
                if (iss >> tau >> kx >> ky >> a >> b >> re >> im) {
                    dataK[{tau,kx,ky,a,b}].first.push_back(re);
                    dataK[{tau,kx,ky,a,b}].second.push_back(im);
                }
            }
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

            // Write to file
            char outnameK[256];
            std::snprintf(outnameK, sizeof(outnameK),
                          "results/%s/statK.dat", filename_.c_str());
            std::ofstream outK(outnameK);
            outK << std::fixed << std::setprecision(12)
                 << std::setw(8)  << "tau"
                 << std::setw(20) << "kx"
                 << std::setw(20) << "ky"
                 << std::setw(4)  << "a"
                 << std::setw(4)  << "b"
                 << std::setw(20) << "re_mean"
                 << std::setw(20) << "re_error"
                 << std::setw(20) << "im_mean"
                 << std::setw(20) << "im_error\n";

            for (const auto& res : resultsK) {
                outK << std::setw(8)  << res.tau
                     << std::setw(20) << res.coord1
                     << std::setw(20) << res.coord2
                     << std::setw(4)  << res.a
                     << std::setw(4)  << res.b
                     << std::setw(20) << res.re_mean
                     << std::setw(20) << res.re_error
                     << std::setw(20) << res.im_mean
                     << std::setw(20) << res.im_error << '\n';
            }
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

namespace Observables {
    double calculate_density(const std::vector<GF>& greens, const Lattice& lat);
    double calculate_doubleOccupancy(const std::vector<GF>& greens, const Lattice& lat);
    double calculate_swavePairing(const std::vector<GF>& greens, const Lattice& lat);
    arma::mat calculate_densityCorr(const std::vector<GF>& greens, const Lattice& lat);
    std::vector<Matrix> calculate_greenTau(const std::vector<GF>& greens, const Lattice& lat);
}

#endif
