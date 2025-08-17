#ifndef HDF5_UTILS_HPP
#define HDF5_UTILS_HPP

#include <armadillo>
#include <string>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <stdexcept>

namespace hdf5 {
    
    inline void write_scalar(hid_t file_id, const std::string& dataset_name, double value) {
        hsize_t dims[1] = {1};
        hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(), H5T_NATIVE_DOUBLE, 
                                      dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                                 H5P_DEFAULT, &value);
        
        if (status < 0) {
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
            throw std::runtime_error("Failed to write scalar to HDF5 file");
        }
        
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
    }
    
    inline void write_matrix(hid_t file_id, const std::string& dataset_name, const arma::mat& matrix) {
        hsize_t dims[2] = {static_cast<hsize_t>(matrix.n_rows), static_cast<hsize_t>(matrix.n_cols)};
        hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(), H5T_NATIVE_DOUBLE, 
                                      dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Create a temporary contiguous copy of the matrix data
        arma::mat temp_matrix = matrix;
        
        herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                                 H5P_DEFAULT, temp_matrix.memptr());
        
        if (status < 0) {
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
            throw std::runtime_error("Failed to write matrix to HDF5 file");
        }
        
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
    }
    
    inline void write_cube(hid_t file_id, const std::string& dataset_name, const arma::cube& cube) {
        hsize_t dims[3] = {static_cast<hsize_t>(cube.n_rows), 
                           static_cast<hsize_t>(cube.n_cols), 
                           static_cast<hsize_t>(cube.n_slices)};
        hid_t dataspace_id = H5Screate_simple(3, dims, NULL);
        hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(), H5T_NATIVE_DOUBLE, 
                                      dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Create a temporary contiguous copy of the cube data
        arma::cube temp_cube = cube;
        
        herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                                 H5P_DEFAULT, temp_cube.memptr());
        
        if (status < 0) {
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
            throw std::runtime_error("Failed to write cube to HDF5 file");
        }
        
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
    }
    
    inline void write_complex_cube(hid_t file_id, const std::string& dataset_name, const arma::cx_cube& cube) {
        hsize_t dims[4] = {static_cast<hsize_t>(cube.n_rows), 
                           static_cast<hsize_t>(cube.n_cols), 
                           static_cast<hsize_t>(cube.n_slices),
                           2};  // 2 for real and imaginary parts
        hid_t dataspace_id = H5Screate_simple(4, dims, NULL);
        hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(), H5T_NATIVE_DOUBLE, 
                                      dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        // Create a temporary contiguous copy of the cube data
        arma::cx_cube temp_cube = cube;
        
        // Convert complex data to interleaved real/imag format for writing
        std::vector<double> temp_data(cube.n_elem * 2);
        const std::complex<double>* cube_data = temp_cube.memptr();
        for (size_t i = 0; i < cube.n_elem; ++i) {
            temp_data[2*i] = cube_data[i].real();     // Real part
            temp_data[2*i + 1] = cube_data[i].imag(); // Imaginary part
        }
        
        herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                                 H5P_DEFAULT, temp_data.data());
        
        if (status < 0) {
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
            throw std::runtime_error("Failed to write complex cube to HDF5 file");
        }
        
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
    }
    
    inline hid_t create_file(const std::string& filename) {
        hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0) {
            throw std::runtime_error("Failed to create HDF5 file: " + filename);
        }
        return file_id;
    }
    
    inline void close_file(hid_t file_id) {
        H5Fclose(file_id);
    }
}

#endif