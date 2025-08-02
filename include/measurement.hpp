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
    arma::field<arma::mat>
    chi_site_to_chi_r(const arma::mat& chi_site,                                    
                    const Lattice&   lat) {                                       
        const int n_orb = lat.n_orb();                         
        const int Lx = lat.Lx();                                                    
        const int Ly = lat.Ly();
        const int n_cells = lat.size();                                            
                                                                                    
        arma::field<arma::mat> chi_r(n_orb, n_orb);                                 
        for (int a = 0; a < n_orb; ++a) {                                          
            for (int b = 0; b < n_orb; ++b) {                                         
                chi_r(a,b).zeros(Lx, Ly);
            }
        }

        for (int ij = 0; ij < static_cast<int>(chi_site.n_elem); ++ij) {
            const int i = ij % chi_site.n_rows;
            const int j = ij / chi_site.n_rows;
            const double val = chi_site(i,j);

            const int a = i % n_orb;
            const int b = j % n_orb;

            const int cell_i = i / n_orb;
            const int cell_j = j / n_orb;

            const int cxi = cell_i % Lx;
            const int cyi = cell_i / Lx;
            const int cxj = cell_j % Lx;
            const int cyj = cell_j / Lx;

            // shortest relative distance under PBC
            int raw_dx = cxj - cxi;
            int dx     = transform::pbc_shortest(raw_dx, Lx);
            int raw_dy = cyj - cyi;
            int dy     = transform::pbc_shortest(raw_dy, Ly);

            // map into [0,Lx-1] and [0,Ly-1] for storage
            dx = dx + Lx/2 - 1;
            dy = dy + Ly/2 - 1;

            chi_r(a,b)(dx, dy) += val / n_cells;
        }

        return chi_r;                                                               
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
            double mean = global_sum_ / global_count_;
            
            std::string outname = filename_ + "_bin.dat";                                                                                                                                                              
            std::ofstream out(outname, std::ios::app); 
            out << std::fixed << std::setprecision(12)
                << std::setw(20) << mean << "\n";
        }

        // reset accumulators
        local_sum_ = 0.0;
        local_count_ = 0;
    }

    void jackknife(int rank) {
        if (rank != 0) return;

        std::string inname = filename_ + "_bin.dat"; 
        std::ifstream in(inname);
        if (!in) {
            std::cerr << "Warning: could not open " << inname << " for jackknife\n";
            return;
        }

        std::vector<double> values;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            double v;
            if (iss >> v) values.push_back(v);
        }
        in.close();

        const auto [means, errors] = statistics::jackknife(values);                                                                                                                                                
        const double mMean  = statistics::mean(means);
        const double mError = statistics::mean(errors);
        
        std::string outname = filename_ + "_stat.dat";                                                                                                                                                              
        std::ofstream out(outname);                                                                                                                                                                                
        out << std::setw(10) << "mean" << std::setw(20) << "error\n";                                                                                                                                              
        out << std::fixed << std::setprecision(12)                                                                                                                                                                 
            << std::setw(10) << mMean << std::setw(20) << mError << '\n';
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

        // convert site matrix → real-space field
        arma::field<arma::mat> chi_r = transform::chi_site_to_chi_r(global_sum / global_count, lat);

        static int bin_counter = 0;
        if (rank == 0) {
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "results/%s/binR_%04d.dat", filename_.c_str(), bin_counter++);

            std::ofstream out(fname);
            out << std::fixed << std::setprecision(12)
                << std::setw(20) << "dx"
                << std::setw(20) << "dy"
                << std::setw(4)  << "a"
                << std::setw(4)  << "b"
                << std::setw(20) << "value" << '\n';

            const int n_orb = chi_r.n_rows;
            const int Lx    = chi_r(0,0).n_rows;
            const int Ly    = chi_r(0,0).n_cols;

            const std::array<double,2>& a1 = lat.a1();
            const std::array<double,2>& a2 = lat.a2();

            for (int rx = 0; rx < Lx; ++rx) {
                for (int ry = 0; ry < Ly; ++ry) {
                    for (int a = 0; a < n_orb; ++a) {
                        for (int b = 0; b < n_orb; ++b) {
                            double dx = (rx - Lx/2 + 1) * a1[0] + (ry - Ly/2 + 1) * a2[0];
                            double dy = (rx - Lx/2 + 1) * a1[1] + (ry - Ly/2 + 1) * a2[1];
                            
                            out << std::setw(20) << dx
                                << std::setw(20) << dy
                                << std::setw(4)  << a
                                << std::setw(4)  << b
                                << std::setw(20) << std::setprecision(12)
                                << chi_r(a,b)(rx,ry) << '\n';
                        }
                    }
                }
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
            std::snprintf(fname, sizeof(fname),                                                                                                                                                                    
                          "results/%s/binR_%04d.dat", filename_.c_str(), idx);                                                                                                                                     
            if (!std::ifstream(fname).good()) break;                                                                                                                                                               
            binsR.emplace_back(fname);                                                                                                                                                                             
        }                                                                                                                                                                                                          
                                                                                                                                                                                                                   
        std::map<std::tuple<double,double,int,int>, std::vector<double>> dataR;                                                                                                                                    
        for (const std::string& bin : binsR) {                                                                                                                                                                     
            std::ifstream in(bin);                                                                                                                                                                                 
            std::string line; std::getline(in, line);          // header                                                                                                                                           
            while (std::getline(in, line)) {                                                                                                                                                                       
                if (line.empty() || line[0] == '#') continue;                                                                                                                                                      
                std::istringstream iss(line);                                                                                                                                                                      
                double dx, dy; int a, b; double v;                                                                                                                                                                 
                if (iss >> dx >> dy >> a >> b >> v)                                                                                                                                                                
                    dataR[{dx,dy,a,b}].push_back(v);                                                                                                                                                               
            }                                                                                                                                                                                                      
        }                                                                                                                                                                                                          
                                                                                                                                                                                                                   
        if (!dataR.empty()) {                                                                                                                                                                                      
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
                                                                                                                                                                                                                   
            for (const auto& [key, vals] : dataR) {                                                                                                                                                                
                if (vals.size() < 2) continue;                                                                                                                                                                     
                const auto [means, errs] = statistics::jackknife(vals);                                                                                                                                            
                const double mMean  = statistics::mean(means);                                                                                                                                                     
                const double mError = statistics::mean(errs);                                                                                                                                                      
                const auto [dx,dy,a,b] = key;                                                                                                                                                                      
                outR << std::setw(20) << dx                                                                                                                                                                        
                     << std::setw(20) << dy                                                                                                                                                                        
                     << std::setw(4)  << a                                                                                                                                                                         
                     << std::setw(4)  << b                                                                                                                                                                         
                     << std::setw(20) << mMean                                                                                                                                                                     
                     << std::setw(20) << mError << '\n';                                                                                                                                                           
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
                                                                                                                                                                                                                   
        if (!dataK.empty()) {                                                                                                                                                                                      
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
                outK << std::setw(20) << kx                                                                                                                                                                        
                     << std::setw(20) << ky                                                                                                                                                                        
                     << std::setw(4)  << a                                                                                                                                                                         
                     << std::setw(4)  << b                                                                                                                                                                         
                     << std::setw(20) << reMean                                                                                                                                                                    
                     << std::setw(20) << reError                                                                                                                                                                   
                     << std::setw(20) << imMean                                                                                                                                                                    
                     << std::setw(20) << imError << '\n';                                                                                                                                                          
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

            // FFT for every orbital pair
            arma::field<arma::cx_mat> chi_k(n_orb, n_orb);
            for (int a = 0; a < n_orb; ++a)
                for (int b = 0; b < n_orb; ++b)
                    chi_k(a,b).set_size(nk, 1);        // store per k

            const std::array<double,2>& a1 = lat.a1();
            const std::array<double,2>& a2 = lat.a2();

            for (int kidx = 0; kidx < nk; ++kidx) {
                const auto& k = kpts[kidx];
                for (int a = 0; a < n_orb; ++a) {
                    for (int b = 0; b < n_orb; ++b) {
                        std::complex<double> sum(0.0, 0.0);
                        for (int rx = 0; rx < lat.Lx(); ++rx) {
                            for (int ry = 0; ry < lat.Ly(); ++ry) {
                                double x = (rx - lat.Lx()/2 + 1) * a1[0]
                                         + (ry - lat.Ly()/2 + 1) * a2[0];
                                double y = (rx - lat.Lx()/2 + 1) * a1[1]
                                         + (ry - lat.Ly()/2 + 1) * a2[1];
                                double phase = k[0]*x + k[1]*y;
                                sum += chi_r(a,b)(rx,ry) *
                                       std::complex<double>(std::cos(phase),
                                                            -std::sin(phase));
                            }
                        }
                        chi_k(a,b)(kidx, 0) = sum;
                    }
                }
                // normalise by number of real-space lattice points
                const double invN = 1.0 / (lat.Lx() * lat.Ly());
                for (int a = 0; a < n_orb; ++a)
                    for (int b = 0; b < n_orb; ++b)
                        chi_k(a,b)(kidx, 0) *= invN;
            }

            // write k-space bin file
            char kname[256];
            std::snprintf(kname, sizeof(kname),
                          "results/%s/binK_%04d.dat", filename_.c_str(), bin_idx);
            std::ofstream kout(kname);
            kout << std::fixed << std::setprecision(12)
                 << std::setw(20) << "kx"
                 << std::setw(20) << "ky"
                 << std::setw(4)  << "a"
                 << std::setw(4)  << "b"
                 << std::setw(20) << "re"
                 << std::setw(20) << "im\n";

            for (int kidx = 0; kidx < nk; ++kidx) {
                const auto& k = kpts[kidx];
                for (int a = 0; a < n_orb; ++a)
                    for (int b = 0; b < n_orb; ++b)
                        kout << std::setw(20) << k[0]
                             << std::setw(20) << k[1]
                             << std::setw(4)  << a
                             << std::setw(4)  << b
                             << std::setw(20) << std::real(chi_k(a,b)(kidx))
                             << std::setw(20) << std::imag(chi_k(a,b)(kidx)) << '\n';
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

    std::vector<arma::mat> local_sum_;  // one matrix per tau slice
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
            local_sum_.resize(n_tau_);
            for (int tau = 0; tau < n_tau_; ++tau) {
                local_sum_[tau].set_size(matrix_size_, matrix_size_);
                local_sum_[tau].zeros();
            }
        }
        
        for (int tau = 0; tau < n_tau_; ++tau) {
            local_sum_[tau] += tau_matrices[tau];
        }
        ++local_count_;
    }

    void accumulate(const Lattice& lat, MPI_Comm comm, int rank) {
        const int n_sites = local_sum_[0].n_rows;
        std::vector<arma::mat> global_sum(n_tau_);
        for (int tau = 0; tau < n_tau_; ++tau) {
            global_sum[tau].set_size(n_sites, n_sites);
            global_sum[tau].zeros();
        }
        int global_count = 0;

        // MPI reduction for each tau slice
        for (int tau = 0; tau < n_tau_; ++tau) {
            MPI_Allreduce(local_sum_[tau].memptr(), global_sum[tau].memptr(),
                          n_sites * n_sites, MPI_DOUBLE, MPI_SUM, comm);
        }
        MPI_Allreduce(&local_count_, &global_count, 1, MPI_INT, MPI_SUM, comm);

        static int bin_counter = 0;
        if (rank == 0) {
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

            const int n_orb = lat.n_orb();
            const int Lx = lat.Lx();
            const int Ly = lat.Ly();
            const std::array<double,2>& a1 = lat.a1();
            const std::array<double,2>& a2 = lat.a2();

            for (int tau = 0; tau < n_tau_; ++tau) {
                // Convert site matrix to real-space field for this tau
                arma::field<arma::mat> chi_r = transform::chi_site_to_chi_r(
                    global_sum[tau] / global_count, lat);

                for (int rx = 0; rx < Lx; ++rx) {
                    for (int ry = 0; ry < Ly; ++ry) {
                        for (int a = 0; a < n_orb; ++a) {
                            for (int b = 0; b < n_orb; ++b) {
                                double dx = (rx - Lx/2 + 1) * a1[0] + (ry - Ly/2 + 1) * a2[0];
                                double dy = (rx - Lx/2 + 1) * a1[1] + (ry - Ly/2 + 1) * a2[1];
                                
                                out << std::setw(8)  << tau
                                    << std::setw(20) << dx
                                    << std::setw(20) << dy
                                    << std::setw(4)  << a
                                    << std::setw(4)  << b
                                    << std::setw(20) << std::setprecision(12)
                                    << chi_r(a,b)(rx,ry) << '\n';
                            }
                        }
                    }
                }
            }
        }

        // Reset accumulators
        for (int tau = 0; tau < n_tau_; ++tau) {
            local_sum_[tau].zeros();
        }
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

            // Write k-space bin file
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
                 << std::setw(20) << "re"
                 << std::setw(20) << "im\n";

            const std::array<double,2>& a1 = lat.a1();
            const std::array<double,2>& a2 = lat.a2();

            // FFT for each tau slice
            for (const auto& [tau, chi_r] : chi_tau_r) {
                for (int kidx = 0; kidx < nk; ++kidx) {
                    const auto& k = kpts[kidx];
                    for (int a = 0; a < n_orb; ++a) {
                        for (int b = 0; b < n_orb; ++b) {
                            std::complex<double> sum(0.0, 0.0);
                            for (int rx = 0; rx < lat.Lx(); ++rx) {
                                for (int ry = 0; ry < lat.Ly(); ++ry) {
                                    double x = (rx - lat.Lx()/2 + 1) * a1[0]
                                             + (ry - lat.Ly()/2 + 1) * a2[0];
                                    double y = (rx - lat.Lx()/2 + 1) * a1[1]
                                             + (ry - lat.Ly()/2 + 1) * a2[1];
                                    double phase = k[0]*x + k[1]*y;
                                    sum += chi_r(a,b)(rx,ry) *
                                           std::complex<double>(std::cos(phase),
                                                                -std::sin(phase));
                                }
                            }
                            // Normalize by number of real-space lattice points
                            sum /= (lat.Lx() * lat.Ly());
                            
                            kout << std::setw(8)  << tau
                                 << std::setw(20) << k[0]
                                 << std::setw(20) << k[1]
                                 << std::setw(4)  << a
                                 << std::setw(4)  << b
                                 << std::setw(20) << std::real(sum)
                                 << std::setw(20) << std::imag(sum) << '\n';
                        }
                    }
                }
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

        if (!dataR.empty()) {
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

            for (const auto& [key, vals] : dataR) {
                if (vals.size() < 2) continue;
                const auto [means, errs] = statistics::jackknife(vals);
                const double mMean = statistics::mean(means);
                const double mError = statistics::mean(errs);
                const auto [tau,dx,dy,a,b] = key;
                outR << std::setw(8)  << tau
                     << std::setw(20) << dx
                     << std::setw(20) << dy
                     << std::setw(4)  << a
                     << std::setw(4)  << b
                     << std::setw(20) << mMean
                     << std::setw(20) << mError << '\n';
            }
        }

        // R0 jackknife analysis
        if (!dataR0.empty()) {
            char outnameR0[256];
            std::snprintf(outnameR0, sizeof(outnameR0),
                          "results/%s/statR0.dat", filename_.c_str());
            std::ofstream outR0(outnameR0);
            outR0 << std::fixed << std::setprecision(12)
                  << std::setw(8)  << "tau"
                  << std::setw(20) << "mean"
                  << std::setw(20) << "error\n";

            for (const auto& [tau, vals] : dataR0) {
                if (vals.size() < 2) continue;
                const auto [means, errs] = statistics::jackknife(vals);
                const double mMean = statistics::mean(means);
                const double mError = statistics::mean(errs);
                outR0 << std::setw(8)  << tau
                      << std::setw(20) << mMean
                      << std::setw(20) << mError << '\n';
            }
        }

        // k-space jackknife analysis
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
                outK << std::setw(8)  << tau
                     << std::setw(20) << kx
                     << std::setw(20) << ky
                     << std::setw(4)  << a
                     << std::setw(4)  << b
                     << std::setw(20) << reMean
                     << std::setw(20) << reError
                     << std::setw(20) << imMean
                     << std::setw(20) << imError << '\n';
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
