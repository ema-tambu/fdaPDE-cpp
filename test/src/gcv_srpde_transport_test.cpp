// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework

#include <cstddef>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::fem_order;
using fdapde::core::FEM;
using fdapde::core::Grid;
using fdapde::core::laplacian;
using fdapde::core::reaction;
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::PDE;
using fdapde::core::DiscretizedVectorField;

#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/gcv.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::SpaceOnly;
using fdapde::models::ExactEDF;
using fdapde::models::GCV;
using fdapde::models::StochasticEDF;
using fdapde::models::Sampling;
#include "../../fdaPDE/calibration/gcv.h"

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_mtx;
using fdapde::testing::read_csv;

using fdapde::core::PDEparameters;


TEST(gcv_srpde_transport_test, transport_testcase0_samplingatlocations_gridexact) {
    // define exact solution
    auto solutionExpr = [](SVector<2> x) -> double {
        return 3*sin(x[0]) + 2*x[1];
    };
    // create noisy data for testing 
    std::default_random_engine generator(123);
    const int n = 100;           // number of observations in each dimension
    const double minVal = 0.0;  // min domain value (unit square)
    const double maxVal = 1.0;  // max domain value (unit square)
    DVector<double> x = DVector<double>::LinSpaced(n, minVal, maxVal);
    DMatrix<double> observations = read_csv<double>("../data/transport/TransportTestCase0/observations" + std::to_string(n) + ".csv");
    DMatrix<double> locs(n*n, 2);   // matrix of spatial locations p_1, p2_, ... p_n
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            locs(i*n + j, 0) = x(i);
            locs(i*n + j, 1) = x(j);
        }
    }
    // add noise to observations (5% of the range of the exact solution values)        
    std::normal_distribution<double> distribution(0.0, 0.05 * (observations.maxCoeff() - observations.minCoeff()));
    DMatrix<double> y(n*n, 1);
    for (int i = 0; i < n*n; ++i) {
        y(i, 0) = observations(i, 0) + distribution(generator);
    }  
    // begin test 
    constexpr std::size_t femOrder = 1;
    // define domain 
    MeshLoader<Mesh2D> domain("unit_square");
    //define regularizing PDE
    SVector<2> b;  b << 1., 1.;
    double nu = 1e-9;
    auto forcingExpr = [&nu, &b](SVector<2> x) -> double {
        return 2*b[1] + 3*b[0]*cos(x[0]) + 3*nu*sin(x[0]);
    };
    ScalarField<2> forcing(forcingExpr);   // wrap lambda expression in ScalarField object
    PDEparameters<decltype(nu), decltype(b)>::destroyInstance();
    PDEparameters<decltype(nu), decltype(b)> &PDEparams =
            PDEparameters<decltype(nu), decltype(b)>::getInstance(nu, b);
    auto L = - nu * laplacian<FEM>() + advection<FEM>(b);
    PDE< decltype(domain.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>, decltype(nu),
            decltype(b)> pde_(domain.mesh, L);
    pde_.set_forcing(forcing);
    pde_.set_stab_param(1.0);
    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    DMatrix<double> solution_ex(nodes_.rows(), 1);
    // set exact sol & dirichlet conditions at PDE level
    for (int i = 0; i < nodes_.rows(); ++i) {
        solution_ex(i) = solutionExpr(nodes_.row(i));
        dirichletBC(i) = solutionExpr(nodes_.row(i));
    }
    pde_.set_dirichlet_bc(dirichletBC);
    // define model
    SRPDE model(pde_, Sampling::pointwise);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    model.init();
    model.set_dirichlet_bc(model.A(), model.b());
    std::cout << "GCV" << std::endl;
    // define GCV function and grid of \lambda_D values
    auto GCV = model.gcv<ExactEDF>();
    std::vector<DVector<double>> lambdas = {SVector<1>(1e-6), SVector<1>(1e-3), SVector<1>(1e-2), SVector<1>(1e-1),
                                            SVector<1>(1.), SVector<1>(5.), SVector<1>(10.), SVector<1>(25.), 
                                            SVector<1>(50.), SVector<1>(1e2), SVector<1>(1e3)};
    // for (double x = -3.0; x <= 3.0; x += 0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
    // optimize GCV
    Grid<fdapde::Dynamic> opt;
    std::cout << "Optimizing GCV..." << std::endl;
    opt.optimize(GCV, lambdas);
    // test correctness
    // EXPECT_TRUE(almost_equal(GCV.edfs(), "../data/models/gcv/2D_test3/edfs.mtx"));
    // EXPECT_TRUE(almost_equal(GCV.gcvs(), "../data/models/gcv/2D_test3/gcvs.mtx"));

    // print results
    auto gcvs = GCV.gcvs();
    for (int i = 0; i < gcvs.size(); ++i) {
        std::cout << "lambda: " << lambdas[i](0) << " \tGCV: " << gcvs[i] << std::endl;
    }
    auto best_lambda = opt.optimum();
    std::cout << "Best lambda: " << best_lambda << std::endl;
    EXPECT_TRUE(1);
}

/*
TEST(gcv_srpde_transport_test, transport_testcase0_samplingatlocations_gridexact_COVARIATES) {
    // define exact solution
    auto solutionExpr = [](SVector<2> x) -> double {
        return 3*sin(x[0]) + 2*x[1];
    };
    // create noisy data for testing 
    std::default_random_engine generator(123);
    const int n = 100;           // number of observations in each dimension
    const double minVal = 0.0;  // min domain value (unit square)
    const double maxVal = 1.0;  // max domain value (unit square)
    DVector<double> x = DVector<double>::LinSpaced(n, minVal, maxVal);
    DMatrix<double> observations = read_csv<double>("../data/transport/TransportTestCase0/observations" + std::to_string(n) + ".csv");
    DMatrix<double> locs(n*n, 2);   // matrix of spatial locations p_1, p2_, ... p_n
    DMatrix<double> X(n*n, 1); // design matrix to add one covariate
    double beta1 = 1.5;    // coefficient for covariate
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            locs(i*n + j, 0) = x(i);
            locs(i*n + j, 1) = x(j);
            X(i*n + j, 0) = x(i)+x(j);  // populate design matrix
        }
    }
    // add noise to observations (5% of the range of the exact solution values)        
    std::normal_distribution<double> distribution(0.0, 0.05 * (observations.maxCoeff() - observations.minCoeff()));
    DMatrix<double> y(n*n, 1);
    for (int i = 0; i < n*n; ++i) {
        // y(i, 0) = observations(i, 0) + distribution(generator);
        y(i, 0) = beta1*X(i, 0) + observations(i, 0) + distribution(generator);  // generate noisy observations with covariate
    }  
    // begin test 
    constexpr std::size_t femOrder = 1;
    // define domain 
    MeshLoader<Mesh2D> domain("unit_square");
    //define regularizing PDE
    SVector<2> b;  b << 1., 1.;
    double nu = 1e-9;
    auto forcingExpr = [&nu, &b](SVector<2> x) -> double {
        return 2*b[1] + 3*b[0]*cos(x[0]) + 3*nu*sin(x[0]);
    };
    ScalarField<2> forcing(forcingExpr);   // wrap lambda expression in ScalarField object
    PDEparameters<decltype(nu), decltype(b)>::destroyInstance();
    PDEparameters<decltype(nu), decltype(b)> &PDEparams =
            PDEparameters<decltype(nu), decltype(b)>::getInstance(nu, b);
    auto L = - nu * laplacian<FEM>() + advection<FEM>(b);
    PDE< decltype(domain.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>, decltype(nu),
            decltype(b)> pde_(domain.mesh, L);
    pde_.set_forcing(forcing);
    pde_.set_stab_param(1.0);
    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    DMatrix<double> solution_ex(nodes_.rows(), 1);
    // set exact sol & dirichlet conditions at PDE level
    for (int i = 0; i < nodes_.rows(); ++i) {
        solution_ex(i) = solutionExpr(nodes_.row(i));
        dirichletBC(i) = solutionExpr(nodes_.row(i));
    }
    pde_.set_dirichlet_bc(dirichletBC);
    // define model
    SRPDE model(pde_, Sampling::pointwise);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    model.init();
    model.set_dirichlet_bc(model.A(), model.b());
    std::cout << "GCV" << std::endl;
    // define GCV function and grid of \lambda_D values
    auto GCV = model.gcv<ExactEDF>();
    std::vector<DVector<double>> lambdas = {SVector<1>(1e-6), SVector<1>(1e-3), SVector<1>(1e-2), SVector<1>(1e-1),
                                            SVector<1>(1.), SVector<1>(5.), SVector<1>(10.), SVector<1>(25.), 
                                            SVector<1>(50.), SVector<1>(1e2), SVector<1>(1e3)};
    // for (double x = -3.0; x <= 3.0; x += 0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
    // optimize GCV
    Grid<fdapde::Dynamic> opt;
    std::cout << "Optimizing GCV..." << std::endl;
    opt.optimize(GCV, lambdas);
    // test correctness
    // EXPECT_TRUE(almost_equal(GCV.edfs(), "../data/models/gcv/2D_test3/edfs.mtx"));
    // EXPECT_TRUE(almost_equal(GCV.gcvs(), "../data/models/gcv/2D_test3/gcvs.mtx"));

    // print results
    auto gcvs = GCV.gcvs();
    for (int i = 0; i < gcvs.size(); ++i) {
        std::cout << "lambda: " << lambdas[i](0) << " \tGCV: " << gcvs[i] << std::endl;
    }
    auto best_lambda = opt.optimum();
    std::cout << "Best lambda: " << best_lambda << std::endl;
    EXPECT_TRUE(1);
}
*/
/*
TEST(gcv_srpde_transport_test, transport_testcase1_samplingatlocations_gridexact) {
    constexpr std::size_t femOrder = 1;
    
    // define PDE coefficients
    SVector<2> b;  b << 1., 0.;
    double nu = 1e-9;
    // define exact solution
    auto solutionExpr = [&nu](SVector<2> x) -> double {
        return x[0]*x[1]*x[1] - x[1]*x[1]*exp((2*(x[0] - 1))/nu) - x[0]*exp(3*(x[1] - 1)/nu) + exp((2*(x[0] - 1) + 3*(x[1] - 1))/nu);
    };
    // forcing term
    using std::exp;
    auto forcingExpr = [&nu, &b](SVector<2> x) -> double {
        return b[0]*(x[1]*x[1] - exp((3*x[1] - 3)/nu) - 2*x[1]*x[1]*exp((2*x[0] - 2)/nu)/nu + 2*exp((2*x[0] + 3*x[1] - 5)/nu)/nu) + b[1]*(2*x[0]*x[1] - 2*x[1]*exp((2*x[0] - 2)/nu) - 3*x[0]*exp((3*x[1] - 3)/nu)/nu + 3*exp((2*x[0] + 3*x[1] - 5)/nu)/nu) - nu*(2*x[0] - 2*exp((2*x[0] - 2)/nu) - 9*x[0]*exp((3*x[1] - 3)/nu)/(nu*nu) - 4*x[1]*x[1]*exp((2*x[0] - 2)/nu)/(nu*nu) + 13*exp((2*x[0] + 3*x[1] - 5)/nu)/(nu*nu));
    };
    ScalarField<2> forcing(forcingExpr);
    // define domain 
    MeshLoader<Mesh2D> domain("unit_square_32");
        
    //define regularizing PDE
    PDEparameters<decltype(nu), decltype(b)>::destroyInstance();
    PDEparameters<decltype(nu), decltype(b)> &PDEparams =
            PDEparameters<decltype(nu), decltype(b)>::getInstance(nu, b);
    auto L = - nu * laplacian<FEM>() + advection<FEM>(b);
    PDE< decltype(domain.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>, decltype(nu),
            decltype(b)> pde_(domain.mesh, L);
    pde_.set_forcing(forcing);
    pde_.set_stab_param(2.075);
    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    DMatrix<double> solution_ex(nodes_.rows(), 1);
    // set exact sol & dirichlet conditions at PDE level
    for (int i = 0; i < nodes_.rows(); ++i) {
        solution_ex(i) = solutionExpr(nodes_.row(i));
        dirichletBC(i) = solutionExpr(nodes_.row(i));
    }
    pde_.set_dirichlet_bc(dirichletBC);

    std::default_random_engine generator(123);
    // create noisy data for testing 
    const int n = 60;           // number of observations in each dimension
    const double minVal = 0.0;  // min domain value (unit square)
    const double maxVal = 1.0;  // max domain value (unit square)
    DVector<double> x = DVector<double>::LinSpaced(n, minVal, maxVal);
    DMatrix<double> locs(n*n, 2);   // matrix of spatial locations p_1, p2_, ... p_n
    DVector<double> eval_sol_ex(n*n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            eval_sol_ex(i*n + j) = solutionExpr(SVector<2>({x(i), x(j)}));
            locs(i*n + j, 0) = x(i);
            locs(i*n + j, 1) = x(j);
        }
    }
    // add noise to observations (5% of the range of the exact solution values)        
    std::normal_distribution<double> distribution(0.0, 0.05 * (eval_sol_ex.maxCoeff() - eval_sol_ex.minCoeff()));
    DMatrix<double> observations(n*n, 1);
    for (int i = 0; i < n*n; ++i) {
        observations(i, 0) = eval_sol_ex(i) + distribution(generator);
    }
    // begin test 
    SRPDE model(pde_, Sampling::pointwise);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, observations);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.set_dirichlet_bc(model.A(), model.b());
    // define GCV function and grid of \lambda_D values
    auto GCV = model.gcv<ExactEDF>();
    std::vector<DVector<double>> lambdas = {SVector<1>(1e-6), SVector<1>(1e-3), SVector<1>(1e-2), SVector<1>(1e-1),
                                            SVector<1>(1.), SVector<1>(5.), SVector<1>(10.), SVector<1>(25.), 
                                            SVector<1>(50.), SVector<1>(1e2), SVector<1>(1e3)};
    // for (double x = -3.0; x <= 3.0; x += 0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
    // optimize GCV
    Grid<fdapde::Dynamic> opt;
    std::cout << "Optimizing GCV..." << std::endl;
    opt.optimize(GCV, lambdas);
    // print results
    auto gcvs = GCV.gcvs();
    for (int i = 0; i < gcvs.size(); ++i) {
        std::cout << "lambda: " << lambdas[i](0) << " \tGCV: " << gcvs[i] << std::endl;
    }
    auto best_lambda = opt.optimum();
    std::cout << "Best lambda: " << best_lambda << std::endl;
    EXPECT_TRUE(1);
}
*/
/*
TEST(gcv_srpde_transport_test, transport_testcase1_samplingatlocations_gridexact_COVARIATES) {
    
    constexpr std::size_t femOrder = 1;
    
    // define PDE coefficients
    SVector<2> b;  b << 1., 0.;
    double nu = 1e-9;
    // define exact solution
    auto solutionExpr = [&nu](SVector<2> x) -> double {
        return x[0]*x[1]*x[1] - x[1]*x[1]*exp((2*(x[0] - 1))/nu) - x[0]*exp(3*(x[1] - 1)/nu) + exp((2*(x[0] - 1) + 3*(x[1] - 1))/nu);
    };
    // forcing term
    using std::exp;
    auto forcingExpr = [&nu, &b](SVector<2> x) -> double {
        return b[0]*(x[1]*x[1] - exp((3*x[1] - 3)/nu) - 2*x[1]*x[1]*exp((2*x[0] - 2)/nu)/nu + 2*exp((2*x[0] + 3*x[1] - 5)/nu)/nu) + b[1]*(2*x[0]*x[1] - 2*x[1]*exp((2*x[0] - 2)/nu) - 3*x[0]*exp((3*x[1] - 3)/nu)/nu + 3*exp((2*x[0] + 3*x[1] - 5)/nu)/nu) - nu*(2*x[0] - 2*exp((2*x[0] - 2)/nu) - 9*x[0]*exp((3*x[1] - 3)/nu)/(nu*nu) - 4*x[1]*x[1]*exp((2*x[0] - 2)/nu)/(nu*nu) + 13*exp((2*x[0] + 3*x[1] - 5)/nu)/(nu*nu));
    };
    ScalarField<2> forcing(forcingExpr);
    
    // define domain 
    MeshLoader<Mesh2D> domain("unit_square_32");
        
    //define regularizing PDE
    PDEparameters<decltype(nu), decltype(b)>::destroyInstance();
    PDEparameters<decltype(nu), decltype(b)> &PDEparams =
            PDEparameters<decltype(nu), decltype(b)>::getInstance(nu, b);
    auto L = - nu * laplacian<FEM>() + advection<FEM>(b);
    PDE< decltype(domain.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>, decltype(nu),
            decltype(b)> pde_(domain.mesh, L);
    pde_.set_forcing(forcing);
    pde_.set_stab_param(1.075);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    DMatrix<double> solution_ex(nodes_.rows(), 1);
    // set exact sol & dirichlet conditions at PDE level
    for (int i = 0; i < nodes_.rows(); ++i) {
        solution_ex(i) = solutionExpr(nodes_.row(i));
        dirichletBC(i) = solutionExpr(nodes_.row(i));
    }
    pde_.set_dirichlet_bc(dirichletBC);

    std::default_random_engine generator(123);
    // create noisy data for testing 
    const int n = 100;           // number of observations in each dimension
    const double minVal = 0.0;  // min domain value (unit square)
    const double maxVal = 1.0;  // max domain value (unit square)
    DVector<double> x = DVector<double>::LinSpaced(n, minVal, maxVal);
    DMatrix<double> locs(n*n, 2);   // matrix of spatial locations p_1, p2_, ... p_n
    DMatrix<double> y(n*n, 1);
    DVector<double> sol_at_locations(n*n, 1); // exact solution at locations
    DMatrix<double> X = DMatrix<double>::Zero(n*n, 1);  // design matrix to add one covariate
    // std::normal_distribution<double> covariate_distribution(0.0, 0.1);
    double beta1 = -1.0;    // coefficient for covariate
    // double beta2 = 2.0;     // coefficient for covariate
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            sol_at_locations(i*n + j) = solutionExpr(SVector<2>({x(i), x(j)}));
            locs(i*n + j, 0) = x(i);
            locs(i*n + j, 1) = x(j);

            X(i*n + j, 0) = locs(i*n + j, 0) * locs(i*n+j, 1);  // populate design matrix
            // X(i*n + j, 1) = covariate_distribution(generator);  
        }
    }
    // add noise to observations (5% of the range of the exact solution values)        
    std::normal_distribution<double> distribution(0.0, 0.05 * (sol_at_locations.maxCoeff() - sol_at_locations.minCoeff()));
    DMatrix<double> observations(n*n, 1);
    for (int i = 0; i < n*n; ++i)
        // y(i, 0) = beta1*X(i, 0) + beta2*X(i,1) + sol_at_locations(i) + distribution(generator);  // generate noisy observations with covariate
        y(i, 0) = beta1*X(i, 0) + sol_at_locations(i) + distribution(generator);  // generate noisy observations with one covariate
    // begin test 
    SRPDE model(pde_, Sampling::pointwise);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.set_dirichlet_bc(model.A(), model.b());
    // define GCV function and grid of \lambda_D values
    auto GCV = model.gcv<ExactEDF>();
    std::vector<DVector<double>> lambdas = {SVector<1>(1e-6), SVector<1>(1e-3), SVector<1>(1e-2), SVector<1>(1e-1),
                                            SVector<1>(1.), SVector<1>(5.), SVector<1>(10.), SVector<1>(25.), 
                                            SVector<1>(50.), SVector<1>(1e2), SVector<1>(1e3)};
    // for (double x = -3.0; x <= 3.0; x += 0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
    // optimize GCV
    Grid<fdapde::Dynamic> opt;
    std::cout << "Optimizing GCV..." << std::endl;
    opt.optimize(GCV, lambdas);
    // print results
    auto gcvs = GCV.gcvs();
    for (int i = 0; i < gcvs.size(); ++i) {
        std::cout << "lambda: " << lambdas[i](0) << " \tGCV: " << gcvs[i] << std::endl;
    }
    auto best_lambda = opt.optimum();
    std::cout << "Best lambda: " << best_lambda << std::endl;
    EXPECT_TRUE(1);
}
*/
/*
TEST(gcv_srpde_transport_test, transport_testcase2_samplingatlocations_gridexact) {
    // for (int n = 40; n <= 120; n+=10){
    constexpr std::size_t femOrder = 1;
    std::default_random_engine generator(123);
    
    // define domain 
    MeshLoader<Mesh2D> domain("unit_square_32");

    // define PDE coefficients
    VectorField<2> b_callable;
    b_callable[0] = [](SVector<2> x) -> double { return std::pow(x[1], 2) + 1; };   // y^2 + 1
    b_callable[1] = [](SVector<2> x) -> double { return 2 * x[0]; };                // 2*x

    fdapde::core::Integrator<FEM, 2, femOrder> integrator;
    DMatrix<double> quad_nodes = integrator.quadrature_nodes(domain.mesh);

    DMatrix<double, Eigen::RowMajor> b_data(quad_nodes.rows(), 2);
    for(int i = 0; i < quad_nodes.rows(); i++) {
        b_data.row(i) = b_callable(SVector<2>(quad_nodes.row(i)));
    }

    // construct b together with its divergence
    ScalarField<2> div_b_callable = div(b_callable);
    DVector<double> div_b_data(quad_nodes.rows());
    for(int i = 0; i < quad_nodes.rows(); i++) {
        div_b_data(i) = div_b_callable(SVector<2>(quad_nodes.row(i)));
    }
    DiscretizedVectorField<2,2> b_discretized(b_data, div_b_data);
    double nu = 1e-9;

    // forcing term
    auto forcingExpr = [](SVector<2> x) -> double { return 1.; };
    ScalarField<2> forcing(forcingExpr);
        
    //define regularizing PDE
    PDEparameters<decltype(nu), decltype(b_discretized)>::destroyInstance();
    PDEparameters<decltype(nu), decltype(b_discretized)> &PDEparams =
            PDEparameters<decltype(nu), decltype(b_discretized)>::getInstance(nu, b_discretized);
    auto L = - nu * laplacian<FEM>() + advection<FEM>(b_discretized);
    PDE< decltype(domain.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>, decltype(nu),
            decltype(b_discretized)> pde_(domain.mesh, L);
    pde_.set_forcing(forcing);
    pde_.set_stab_param(2.285);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    // set dirichlet conditions at PDE level
    for (int i = 0; i < nodes_.rows(); ++i) {
        dirichletBC(i) = 0.; // solutionExpr(nodes_.row(i));
    }
    pde_.set_dirichlet_bc(dirichletBC);
    
    // create noisy data for testing 
    const int n = 60;           // number of observations in each dimension
    const double minVal = 0.0;  // min domain value (unit square)
    const double maxVal = 1.0;  // max domain value (unit square)

    DVector<double> x = DVector<double>::LinSpaced(n, minVal, maxVal);
    DMatrix<double> locs(n*n, 2);   // matrix of spatial locations p_1, p2_, ... p_n
    DMatrix<double> observations = read_csv<double>("../data/transport/TransportTestCase2/observations" + std::to_string(n) + ".csv");
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            locs(i*n + j, 0) = x(i);
            locs(i*n + j, 1) = x(j);
        }
    }
    // add noise to observations (5% of the range of the exact solution values)        
    std::normal_distribution<double> distribution(0.0, 0.05 * (observations.maxCoeff() - observations.minCoeff()));
    
    DMatrix<double> y(n*n, 1);
    for (int i = 0; i < n*n; ++i)
        // y(i, 0) = beta1*X(i, 0) + beta2*X(i,1) + sol_at_locations(i) + distribution(generator);  // generate noisy observations with covariate
        y(i, 0) = observations(i) + distribution(generator);

    // begin test 
    SRPDE model(pde_, Sampling::pointwise);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.set_dirichlet_bc(model.A(), model.b());
    // define GCV function and grid of \lambda_D values
    auto GCV = model.gcv<ExactEDF>();
    std::vector<DVector<double>> lambdas = {SVector<1>(1e-6), SVector<1>(1e-3), SVector<1>(1e-2), SVector<1>(1e-1),
                                            SVector<1>(1.), SVector<1>(5.), SVector<1>(10.), SVector<1>(25.), 
                                            SVector<1>(50.), SVector<1>(1e2), SVector<1>(1e3)};
    // for (double x = -3.0; x <= 3.0; x += 0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
    // optimize GCV
    Grid<fdapde::Dynamic> opt;
    std::cout << "Optimizing GCV..." << std::endl;
    opt.optimize(GCV, lambdas);
    // print results
    auto gcvs = GCV.gcvs();
    for (int i = 0; i < gcvs.size(); ++i) {
        std::cout << "lambda: " << lambdas[i](0) << " \tGCV: " << gcvs[i] << std::endl;
    }
    auto best_lambda = opt.optimum();
    std::cout << "Best lambda: " << best_lambda << std::endl;

    EXPECT_TRUE(1);
}
*/
TEST(gcv_srpde_transport_test, transport_testcase3_samplingatlocations_gridexact) {

    // for (int n = 35; n <= 180; n += 5){
    // std::cout << "n: " << n << std::endl;
    int n = 100;

    constexpr std::size_t femOrder = 1;
    std::default_random_engine generator(123);
    
    // define domain 
    MeshLoader<Mesh2D> domain("unit_square_32");

    // define PDE coefficients
    VectorField<2> b_callable;
    b_callable[0] = [](SVector<2> x) -> double { return std::pow(x[1], 2) + 1; };   // y^2 + 1
    b_callable[1] = [](SVector<2> x) -> double { return 2 * x[0]; };                // 2*x

    fdapde::core::Integrator<FEM, 2, femOrder> integrator;
    DMatrix<double> quad_nodes = integrator.quadrature_nodes(domain.mesh);

    DMatrix<double, Eigen::RowMajor> b_data(quad_nodes.rows(), 2);
    for(int i = 0; i < quad_nodes.rows(); i++) {
        b_data.row(i) = b_callable(SVector<2>(quad_nodes.row(i)));
    }

    // construct b together with its divergence
    ScalarField<2> div_b_callable = div(b_callable);
    DVector<double> div_b_data(quad_nodes.rows());
    for(int i = 0; i < quad_nodes.rows(); i++) {
        div_b_data(i) = div_b_callable(SVector<2>(quad_nodes.row(i)));
    }
    DiscretizedVectorField<2,2> b_discretized(b_data, div_b_data);
    double nu = 1e-5;
    double c = 1.0;

    // forcing term
    auto forcingExpr = [](SVector<2> x) -> double { return 1.; };
    ScalarField<2> forcing(forcingExpr);
        
    // forward coefficients to PDEparams
    PDEparameters<decltype(nu), decltype(b_discretized), decltype(c)>::destroyInstance();
    PDEparameters<decltype(nu), decltype(b_discretized), decltype(c)> &PDEparams =
            PDEparameters<decltype(nu), decltype(b_discretized), decltype(c)>::getInstance(nu, b_discretized, c);
    
    auto L = - nu * laplacian<FEM>() + advection<FEM>(b_discretized) + reaction<FEM>(c);
    
    PDE< decltype(domain.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>, decltype(nu),
            decltype(b_discretized), decltype(c)> pde_(domain.mesh, L);
    pde_.set_forcing(forcing);
    pde_.set_stab_param(2.285);

    // set dirichlet conditions at PDE level
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    for (int i = 0; i < nodes_.rows(); ++i) {
        dirichletBC(i) = 0.; // solutionExpr(nodes_.row(i));
    }
    pde_.set_dirichlet_bc(dirichletBC);

    // prepare data for testing 
    const double minVal = 0.0;  // min domain value (unit square)
    const double maxVal = 1.0;  // max domain value (unit square)
    DVector<double> x = DVector<double>::LinSpaced(n, minVal, maxVal);
    DMatrix<double> locs(n*n, 2);   // matrix of spatial locations p_1, p2_, ... p_n
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            locs(i*n + j, 0) = x(i);
            locs(i*n + j, 1) = x(j);
        }
    }
    // add noise to observations (5% of the range of the exact solution values)        
    DMatrix<double> observations = read_csv<double>("../data/transport/TransportTestCase3/observations" + std::to_string(n) + ".csv");
    std::normal_distribution<double> distribution(0.0, 0.05 * (observations.maxCoeff() - observations.minCoeff()));        
    DMatrix<double> y(n*n, 1);
    for (int i = 0; i < n*n; ++i)
        y(i, 0) = observations(i) + distribution(generator);

    // begin test 
    SRPDE model(pde_, Sampling::pointwise);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.set_dirichlet_bc(model.A(), model.b());
    // define GCV function and grid of \lambda_D values
    auto GCV = model.gcv<ExactEDF>();
    std::vector<DVector<double>> lambdas = {SVector<1>(1e-6), SVector<1>(1e-3), SVector<1>(1e-2), SVector<1>(1e-1),
                                            SVector<1>(1.), SVector<1>(5.), SVector<1>(10.), SVector<1>(25.), 
                                            SVector<1>(50.), SVector<1>(1e2), SVector<1>(1e3)};
    // for (double x = -3.0; x <= 3.0; x += 0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
    // optimize GCV
    Grid<fdapde::Dynamic> opt;
    std::cout << "Optimizing GCV..." << std::endl;
    opt.optimize(GCV, lambdas);
    // print results
    auto gcvs = GCV.gcvs();
    for (int i = 0; i < gcvs.size(); ++i) {
        std::cout << "lambda: " << lambdas[i](0) << " \tGCV: " << gcvs[i] << std::endl;
    }
    auto best_lambda = opt.optimum();
    std::cout << "Best lambda: " << best_lambda << std::endl;

    EXPECT_TRUE(1);
}
/*
TEST(gcv_srpde_transport_test, transport_testcase5_samplingatlocations_gridexact) {
    // for (int n = 40; n <= 120; n+=10){
    constexpr std::size_t femOrder = 1;
    std::default_random_engine generator(123);
    
    // define domain 
    MeshLoader<Mesh2D> domain("nonConvex1");

    // define PDE coefficients
    VectorField<2> b_callable;
    b_callable[0] = [](SVector<2> x) -> double { return std::log(x[0] + 5); };   // log(x+5)
    b_callable[1] = [](SVector<2> x) -> double { return -(x[1] + 1); };          // -(y+1)

    fdapde::core::Integrator<FEM, 2, femOrder> integrator;
    DMatrix<double> quad_nodes = integrator.quadrature_nodes(domain.mesh);

    DMatrix<double, Eigen::RowMajor> b_data(quad_nodes.rows(), 2);
    for(int i = 0; i < quad_nodes.rows(); i++) {
        b_data.row(i) = b_callable(SVector<2>(quad_nodes.row(i)));
    }

    // construct b together with its divergence
    ScalarField<2> div_b_callable = div(b_callable);
    DVector<double> div_b_data(quad_nodes.rows());
    for(int i = 0; i < quad_nodes.rows(); i++) {
        div_b_data(i) = div_b_callable(SVector<2>(quad_nodes.row(i)));
    }
    DiscretizedVectorField<2,2> b_discretized(b_data, div_b_data);
    double nu = 1e-3;

    // forcing term
    auto forcingExpr = [](SVector<2> x) -> double { return 1.; };
    ScalarField<2> forcing(forcingExpr);
        
    // prepare boundary matrix for neumann boundary conditions
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);
    DVector<int> NeumannNodes(54);
    NeumannNodes << 2, 6, 13, 24, 40, 60, 81, 102, 126, 151, 175, 200, 227, 252, 278, 303, 317, 340, 363, 370, 393, 414, 435, 441, 462, 484, 505, 524, 543, 563, 581, 600, 616, 632, 648, 661, 671, 681, 692, 700, 708, 715, 721, 727, 733, 740, 745, 746, 750, 751, 754, 755, 756, 757;
    for (size_t j=0; j<NeumannNodes.size(); ++j){
        int i = NeumannNodes(j);
        boundary_matrix(i, 0) = static_cast<short int>(1);
    }
    
    // forward PDE coefficients
    PDEparameters<decltype(nu), decltype(b_discretized)>::destroyInstance();
    PDEparameters<decltype(nu), decltype(b_discretized)> &PDEparams =
            PDEparameters<decltype(nu), decltype(b_discretized)>::getInstance(nu, b_discretized);
    
    // define differential operator
    auto L = - nu * laplacian<FEM>() + advection<FEM>(b_discretized);

    // define PDE
    PDE< decltype(domain.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>, decltype(nu),
            decltype(b_discretized)> pde_(domain.mesh, L, boundary_matrix);
    pde_.set_forcing(forcing);

    // set dirichlet conditions at PDE level
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    for (int i = 0; i < nodes_.rows(); ++i) {
        dirichletBC(i) = 0.; // solutionExpr(nodes_.row(i));
    }
    pde_.set_dirichlet_bc(dirichletBC);

    // set also neumann b.c.
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), 1);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        f_neumann(i) = 0;
    }
    pde_.set_neumann_bc(f_neumann);

    pde_.set_stab_param(5.0);

    int n = 250;
    
    // import matrix of spatial locations p_1, p2_, ... p_n
    DMatrix<double> locs = read_csv<double>("../data/transport/TransportTestCase6/locs" + std::to_string(n) + ".csv");
    
    // import matrix of observations
    DMatrix<double> observations = read_csv<double>("../data/transport/TransportTestCase6/observations" + std::to_string(n) + ".csv");

    // add noise to observations (5% of the range of the exact solution values)        
    std::normal_distribution<double> distribution(0.0, 0.05 * (observations.maxCoeff() - observations.minCoeff()));
    DMatrix<double> y(n, 1);
    for (int i = 0; i < n; ++i)
        y(i, 0) = observations(i) + distribution(generator);

    // begin test 
    SRPDE model(pde_, Sampling::pointwise);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.set_dirichlet_bc(model.A(), model.b());
    // define GCV function and grid of \lambda_D values
    auto GCV = model.gcv<ExactEDF>();
    std::vector<DVector<double>> lambdas = {SVector<1>(1e-6), SVector<1>(1e-3), SVector<1>(1e-2), SVector<1>(1e-1),
                                            SVector<1>(1.), SVector<1>(5.), SVector<1>(10.), SVector<1>(25.), 
                                            SVector<1>(50.), SVector<1>(1e2), SVector<1>(1e3)};
    // for (double x = -3.0; x <= 3.0; x += 0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
    // optimize GCV
    Grid<fdapde::Dynamic> opt;
    std::cout << "Optimizing GCV..." << std::endl;
    opt.optimize(GCV, lambdas);
    // print results
    auto gcvs = GCV.gcvs();
    for (int i = 0; i < gcvs.size(); ++i) {
        std::cout << "lambda: " << lambdas[i](0) << " \tGCV: " << gcvs[i] << std::endl;
    }
    auto best_lambda = opt.optimum();
    std::cout << "Best lambda: " << best_lambda << std::endl;

    EXPECT_TRUE(1);
}
*/