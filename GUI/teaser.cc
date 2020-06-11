// An example showing TEASER++ registration with the Stanford bunny model
#include <chrono>
#include <iostream>
#include <random>

#include <Eigen/Core>

#include <teaser/ply_io.h>
#include <teaser/registration.h>


int main(int argc, char* argv[]) {

  // argv[1:]: "path_to_src_cloud", "path_to_out_cloud", "float_noise_bound", "float.cbar2", "bool_estimate_scaling",
  //            "int_rotation_max_iterations", "float_rotation_gnc_factor", "float_rotation_cost_threshold"

  // Load the .ply file
  teaser::PLYReader reader;
  teaser::PointCloud src_cloud, tgt_cloud;
  auto status = reader.read(argv[1], src_cloud);
  status = reader.read(argv[2], tgt_cloud);

  int N_src = src_cloud.size();
  int N_tgt = tgt_cloud.size();

  // Convert the point clouds to Eigen
  Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, N_src);
  for (size_t i = 0; i < N_src; ++i) {
    src.col(i) << src_cloud[i].x, src_cloud[i].y, src_cloud[i].z;
  }
  Eigen::Matrix<double, 3, Eigen::Dynamic> tgt(3, N_tgt);
  for (size_t i = 0; i < N_tgt; ++i) {
    tgt.col(i) << tgt_cloud[i].x, tgt_cloud[i].y, tgt_cloud[i].z;
  }

  // Run TEASER++ registration
  // Prepare solver parameters
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = std::stof(argv[3]);
  params.cbar2 = std::stof(argv[4]);
  params.estimate_scaling = std::stoi(argv[5]);
  params.rotation_max_iterations = std::stoi(argv[6]);
  params.rotation_gnc_factor = std::stof(argv[7]);
  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  params.rotation_cost_threshold = std::stof(argv[8]);

  // Solve with TEASER++
  teaser::RobustRegistrationSolver solver(params);
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  solver.solve(src, tgt);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  auto solution = solver.getSolution();

  // Compare results
  std::cout << "=====================================" << std::endl;
  std::cout << "          TEASER++ Results           " << std::endl;
  std::cout << "=====================================" << std::endl;
    std::cout << "Estimated rotation: " << std::endl;
  std::cout << solution.rotation << std::endl;
  std::cout << "Estimated translation: " << std::endl;
  std::cout << solution.translation << std::endl;
  std::cout << "Estimated scale: " << std::endl;
  std::cout << solution.scale << std::endl;
  std::cout << std::endl;

  std::cout << "Time taken (s): "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
                   1000000.0
            << std::endl;
}
