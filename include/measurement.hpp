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
#include <sys/stat.h>

class scalarObservable {
private:
    std::string filename_;
    
    // accumulation
    double local_sum_ = 0.0;
    double global_sum_ = 0.0;

    int local_count_ = 0;
    int global_count_ = 0;

    bool ensure_results_dir(int rank) {
        if (rank == 0) {
            struct stat info;
            if (stat("results", &info) != 0) {
                #ifdef _WIN32
                    int status = _mkdir("results");
                #else
                    int status = mkdir("results", 0755);
                #endif
                if (status != 0) return false;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
    }

public:
    scalarObservable(const std::string& filename, int rank) : filename_("results/" + filename) {
        if (!ensure_results_dir(rank)) {
            throw std::runtime_error("Could not create results directory");
        }
        
        if (rank == 0) {
            std::ofstream out(filename_);
            out << "#" << std::setw(20) << "value\n" ;
        }
    }

    const std::string& filename() const { return filename_; }
    
    void operator+=(double value) {
        local_sum_ += value;
        local_count_++;
    }
    
    void accumulate(MPI_Comm comm) {
        MPI_Allreduce(&local_sum_, &global_sum_, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_count_, &global_count_, 1, MPI_INT, MPI_SUM, comm);
        
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            double mean = global_sum_ / global_count_;
            
            std::ofstream out(filename_, std::ios::app);
            out << std::fixed << std::setprecision(10)
                << std::setw(20) << mean << "\n";
        }

        // reset accumulators
        local_sum_ = 0.0;
        local_count_ = 0;
    }

    static void jackknife(const std::string& filename,
                          MPI_Comm comm,
                          int rank) {
        if (rank != 0) return;

        std::ifstream in(filename);
        if (!in) {
            std::cerr << "Warning: could not open " << filename << " for jackknife\n";
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

        const size_t N = values.size();
        if (N == 0) {
            std::cerr << "Warning: no data in " << filename << "\n";
            return;
        }

        // Jackknife resampling
        std::vector<double> jack_samples(N);
        for (size_t k = 0; k < N; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < N; ++j) {
                if (j != k) sum += values[j];
            }
            jack_samples[k] = sum / (N - 1);
        }

        // Compute per-sample jackknife error
        std::vector<double> jack_errors(N);
        for (size_t k = 0; k < N; ++k) {
            double sum_sq = 0.0;
            double sum    = 0.0;
            for (size_t j = 0; j < N; ++j) {
                if (j == k) continue;
                sum    += values[j];
                sum_sq += values[j] * values[j];
            }
            const double m   = sum / (N - 1);
            const double var = (sum_sq / (N - 1)) - (m * m);
            jack_errors[k]   = std::sqrt(var / (N - 2));
        }

        // Overwrite file with resamples + individual errors
        std::ofstream out(filename, std::ios::trunc);
        out << "#" << std::setw(10) << "index"
            << std::setw(20) << "resample mean"
            << std::setw(20) << "error\n";
        for (size_t k = 0; k < N; ++k) {
            out << std::setw(10) << (k + 1)
                << std::fixed << std::setprecision(10)
                << std::setw(20) << jack_samples[k]
                << std::setw(20) << jack_errors[k] << "\n";
        }
    }
    
};

class equalTimeObservable {
private:
    std::string name_;
    int rank_;
    int matrix_size_ = 0;  // inferred on first matrix

    arma::mat local_sum_;
    int local_count_ = 0;

    bool ensure_obs_dir() {
        if (rank_ == 0) {
            struct stat info;
            std::string path = "results/" + name_;
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

public:
    equalTimeObservable(const std::string& name, int rank)
        : name_(name), rank_(rank) {
        if (!ensure_obs_dir()) {
            throw std::runtime_error("Could not create results/" + name_ + " directory");
        }
    }

    const std::string& name() const { return name_; }

    void operator+=(const arma::mat& m) {
        if (matrix_size_ == 0) {
            matrix_size_ = m.n_rows;
            local_sum_.set_size(matrix_size_, matrix_size_);
            local_sum_.zeros();
        }
        local_sum_ += m;
        ++local_count_;
    }

    void accumulate(MPI_Comm comm) {
        arma::mat global_sum(matrix_size_, matrix_size_, arma::fill::zeros);
        int global_count = 0;

        MPI_Allreduce(local_sum_.memptr(), global_sum.memptr(),
                      matrix_size_ * matrix_size_, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_count_, &global_count, 1, MPI_INT, MPI_SUM, comm);

        static int bin_counter = 0;
        if (rank_ == 0) {
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "results/%s/bin_%06d.dat", name_.c_str(), bin_counter++);
            std::ofstream out(fname);
            const unsigned N = global_sum.n_rows;               // full side length
            const unsigned start = N / 2;                       // first upper-triangle index
            const unsigned nUpper = N - start;                  // elements per side in upper block
            out << "# nk " << nUpper * nUpper << '\n';
            out << "# i j value\n";
            out << std::fixed << std::setprecision(10);
            for (unsigned i = start; i < N; ++i) {
                for (unsigned j = start; j < N; ++j) {
                    out << i << ' ' << j << ' '
                        << global_sum(i, j) / global_count << '\n';
                }
            }
        }

        // reset accumulators
        local_sum_.zeros();
        local_count_ = 0;
    }

    static void jackknife(const std::string& name,
                          MPI_Comm comm,
                          int rank) {
        if (rank != 0) return;

        // 1. discover how many (i,j) pairs we have
        int nk = 0;
        {
            std::ifstream tmp;
            char fname[256];
            std::snprintf(fname, sizeof(fname), "results/%s/bin_000000.dat", name.c_str());
            tmp.open(fname);
            if (!tmp) return;
            std::string line;
            std::getline(tmp, line);            // "# nk <value>"
            std::istringstream iss(line);
            std::string dummy; iss >> dummy >> dummy >> nk;
        }
        if (nk <= 0) return;

        // 2. collect per-(i,j) vectors
        std::vector<std::vector<double>> values(nk);
        int bin_idx = 0;
        while (true) {
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "results/%s/bin_%06d.dat", name.c_str(), bin_idx);
            std::ifstream in(fname);
            if (!in) break;
            std::string line;
            std::getline(in, line);            // skip header
            for (int k = 0; k < nk; ++k) {
                std::getline(in, line);
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                int i, j; double v; iss >> i >> j >> v;
                values[k].push_back(v);
            }
            ++bin_idx;
        }

        // 3. write final file: i j mean error
        char jk_fname[256];
        std::snprintf(jk_fname, sizeof(jk_fname), "results/%s/final.jk.dat", name.c_str());
        std::ofstream out(jk_fname);
        out << "#" << std::setw(10) << "i"
            << std::setw(10) << "j"
            << std::setw(20) << "mean"
            << std::setw(20) << "error\n";
        out << std::fixed << std::setprecision(10);

        for (int k = 0; k < nk; ++k) {
            if (values[k].empty()) continue;

            // temporary file name for scalar routine
            char tmp[256];
            std::snprintf(tmp, sizeof(tmp), "results/%s/_tmp_%d.dat", name.c_str(), k);
            {
                std::ofstream tmp_out(tmp);
                tmp_out << std::fixed << std::setprecision(10);
                for (double v : values[k]) tmp_out << v << '\n';
            }

            // run scalar jack-knife on this subset
            scalarObservable::jackknife(tmp, comm, rank);

            // read back the single jack-knife result
            std::ifstream res(tmp);
            std::string line;
            while (std::getline(res, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                int idx; double mean, err; iss >> idx >> mean >> err;
                // retrieve (i,j) from the original data structure
                // For equal time observables, we need to track the original indices
                // Since we can't reconstruct them from the values alone, we'll use
                // the fact that k corresponds to the (i,j) pair order in the first bin file
                int nk = 0;
                {
                    std::ifstream tmp_bin;
                    char fname[256];
                    std::snprintf(fname, sizeof(fname), "results/%s/bin_000000.dat", name.c_str());
                    tmp_bin.open(fname);
                    if (tmp_bin) {
                        std::string line;
                        std::getline(tmp_bin, line); // skip first header
                        for (int k2 = 0; k2 <= k; ++k2) {
                            std::getline(tmp_bin, line);
                        }
                        std::istringstream iss(line);
                        int i, j; double v; iss >> i >> j >> v;
                        out << std::setw(10) << i
                            << std::setw(10) << j
                            << std::setw(20) << mean
                            << std::setw(20) << err << '\n';
                    }
                }
                break;   // one line per k
            }
            std::remove(tmp);
        }

        // delete original bin files
        bin_idx = 0;
        while (true) {
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "results/%s/bin_%06d.dat", name.c_str(), bin_idx);
            if (std::remove(fname) != 0) break;
            ++bin_idx;
        }
    }
};

class MeasurementManager {
private:
    std::vector<scalarObservable> scalarObservables_;
    std::vector<std::function<double(const std::vector<GF>&)>> calculators_;
    MPI_Comm comm_;
    int rank_;

    std::vector<equalTimeObservable> equalTimeObservables_;
    std::vector<std::function<arma::mat(const std::vector<GF>&)>> eqTimeCalculators_;
    
public:
    MeasurementManager(MPI_Comm comm, int rank) : comm_(comm), rank_(rank) {}

    void addScalar(const std::string& name, 
             std::function<double(const std::vector<GF>&)> calculator) {
        scalarObservables_.emplace_back(name + ".dat", rank_);
        calculators_.push_back(calculator);
    }

    void addEqualTime(const std::string& name,
                      std::function<arma::mat(const std::vector<GF>&)> calculator) {
        equalTimeObservables_.emplace_back(name, rank_);
        eqTimeCalculators_.push_back(calculator);
    }

    void measure(const std::vector<GF>& greens) {
        for (size_t i = 0; i < calculators_.size(); ++i) {
            scalarObservables_[i] += calculators_[i](greens);
        }
        for (size_t i = 0; i < eqTimeCalculators_.size(); ++i) {
            equalTimeObservables_[i] += eqTimeCalculators_[i](greens);
        }
    }

    void accumulate() {
        for (auto& obs : scalarObservables_) {
            obs.accumulate(comm_);
        }
        for (auto& obs : equalTimeObservables_) {
            obs.accumulate(comm_);
        }
    }


    // jackknife analysis
    void jacknifeAnalysis() {
        for (auto& obs : scalarObservables_) {
            scalarObservable::jackknife(obs.filename(), comm_, rank_);
        }
        for (auto& obs : equalTimeObservables_) {
            equalTimeObservable::jackknife(obs.name(), comm_, rank_);
        }
        MPI_Barrier(comm_);
    }
};

namespace Observables {
    double calculate_density(const std::vector<GF>& greens);
    double calculate_doubleOccupancy(const std::vector<GF>& greens);
    double calculate_swavePairing(const std::vector<GF>& greens);
    arma::mat calculate_densityCorr(const std::vector<GF>& greens);
}

#endif
