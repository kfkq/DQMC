#ifndef MEASUREMENT_HPP
#define MEASUREMENT_HPP

#include "dqmc.hpp"

#include <mpi.h>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <map>
#include <tuple>
#include <sys/stat.h>

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

static bool ensure_dir(const std::string& path, int rank)
{
    if (rank == 0) {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
#ifdef _WIN32
            int status = _mkdir(path.c_str());
#else
            int status = mkdir(path.c_str(), 0755);
#endif
            if (status != 0) return false;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
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
        if (!ensure_dir("results", rank)) {
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
        if (!ensure_dir("results/" + filename_, rank)) {
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
                          "results/%s/bin_%04d.dat", filename_.c_str(), bin_counter++);

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
                                                                                                                                                                                                                    
        /* 1. collect all bin files */                                                                                                                                                                                 
        std::vector<std::string> bins;                                                                                                                                                                                 
        for (int idx = 0; ; ++idx) {                                                                                                                                                                                   
            char fname[256];                                                                                                                                                                                           
            std::snprintf(fname, sizeof(fname), "results/%s/bin_%04d.dat", filename_.c_str(), idx);                                                                                                                         
            if (!std::ifstream(fname).good()) break;                                                                                                                                                                   
            bins.emplace_back(fname);                                                                                                                                                                                  
        }                                                                                                                                                                                                              
        if (bins.empty()) return;                                                                                                                                                                                      
                                                                                                                                                                                                                    
        /* 2. build vectors: key = (dx,dy,a,b) -> values over bins */                                                                                                                                                  
        std::map<std::tuple<double,double,int,int>, std::vector<double>> data;                                                                                                                                         
                                                                                                                                                                                                                    
        for (const std::string& bin : bins) {                                                                                                                                                                          
            std::ifstream in(bin);                                                                                                                                                                                     
            std::string line; std::getline(in, line);        // skip header                                                                                                                                            
            while (std::getline(in, line)) {                                                                                                                                                                           
                if (line.empty() || line[0] == '#') continue;                                                                                                                                                          
                std::istringstream iss(line);                                                                                                                                                                          
                double dx, dy; int a, b; double v;                                                                                                                                                                     
                if (iss >> dx >> dy >> a >> b >> v)                                                                                                                                                                    
                    data[{dx,dy,a,b}].push_back(v);                                                                                                                                                                    
            }                                                                                                                                                                                                          
        }                                                                                                                                                                                                              
                                                                                                                                                                                                                    
        /* 3. run jackknife per key and write results */                                                                                                                                                               
        char outname[256];                                                                                                                                                                                             
        std::snprintf(outname, sizeof(outname), "results/%s/stat.dat", filename_.c_str());                                                                                                                         
        std::ofstream out(outname);                                                                                                                                                                                    
        out << std::fixed << std::setprecision(12)
            << std::setw(20) << "dx"                                                                                                                                                                            
            << std::setw(20) << "dy"                                                                                                                                                                                   
            << std::setw(4)  << "a"                                                                                                                                                                                    
            << std::setw(4)  << "b"                                                                                                                                                                                    
            << std::setw(20) << "mean"                                                                                                                                                                                 
            << std::setw(20) << "error\n";                                                                                                                                                                             
        out << std::fixed << std::setprecision(12);                                                                                                                                                                    
                                                                                                                                                                                                                    
        for (const auto& [key, vals] : data) {                                                                                                                                                                         
            if (vals.size() < 2) continue;                                                                                                                                                                             
                                                                                                                                                                                                                    
            const auto [means, errs]  = statistics::jackknife(vals);                                                                                                                                                   
            const double mMean  = statistics::mean(means);                                                                                                                                                             
            const double mError = statistics::mean(errs);                                                                                                                                                              
                                                                                                                                                                                                                    
            const auto [dx,dy,a,b] = key;                                                                                                                                                                              
            out << std::setw(20) << dx                                                                                                                                                                                 
                << std::setw(20) << dy                                                                                                                                                                                 
                << std::setw(4)  << a                                                                                                                                                                                  
                << std::setw(4)  << b                                                                                                                                                                                  
                << std::setw(20) << mMean                                                                                                                                                                              
                << std::setw(20) << mError << '\n';                                                                                                                                                                    
        }                                                                                                                                                
    } 
};

class MeasurementManager {
private:                                                                                                                                                                                                           
    std::vector<scalarObservable> scalarObservables_;                                                                                                                                                              
    std::vector<std::function<double(const std::vector<GF>&, const Lattice&)>> calculators_;                                                                                                                       
    MPI_Comm comm_;                                                                                                                                                                                                
    int rank_;                                                                                                                                                                                                     
                                                                                                                                                                                                                   
    std::vector<equalTimeObservable> equalTimeObservables_;                                                                                                                                                        
    std::vector<std::function<arma::mat(const std::vector<GF>&, const Lattice&)>> eqTimeCalculators_; 
    
public:
    MeasurementManager(MPI_Comm comm, int rank) : comm_(comm), rank_(rank) {}

    void addScalar(const std::string& name, 
             std::function<double(const std::vector<GF>&, const Lattice& lat)> calculator) {
        scalarObservables_.emplace_back(name, rank_);
        calculators_.push_back(calculator);
    }

    void addEqualTime(const std::string& name,
                      std::function<arma::mat(const std::vector<GF>&, const Lattice& lat)> calculator) {
        equalTimeObservables_.emplace_back(name, rank_);
        eqTimeCalculators_.push_back(calculator);
    }

    void measure(const std::vector<GF>& greens, const Lattice& lat) {
        for (size_t i = 0; i < calculators_.size(); ++i) {
            scalarObservables_[i] += calculators_[i](greens, lat);
        }
        for (size_t i = 0; i < eqTimeCalculators_.size(); ++i) {
            equalTimeObservables_[i] += eqTimeCalculators_[i](greens, lat);
        }
    }

    void accumulate(const Lattice& lat) {
        for (auto& obs : scalarObservables_) {
            obs.accumulate(comm_, rank_);
            
        }
        for (auto& obs : equalTimeObservables_) {
            obs.accumulate(lat, comm_, rank_);
        }
    }


    // jackknife analysis
    void jacknifeAnalysis() {
        for (auto& obs : scalarObservables_) {
            obs.jackknife(rank_);
        }
        for (auto& obs : equalTimeObservables_) {
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
}

#endif
