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
using fdapde::core::fem_order;
using fdapde::core::FEM;
using fdapde::core::laplacian;
using fdapde::core::PDE;

#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/qsrpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::QSRPDE;
using fdapde::models::SpaceOnly;
using fdapde::models::Sampling;
#include "../../fdaPDE/calibration/kfold_cv.h"
#include "../../fdaPDE/calibration/rmse.h"
using fdapde::calibration::KCV;
using fdapde::calibration::RMSE;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_mtx;
using fdapde::testing::read_csv;


using fdapde::core::PDEparameters;
using fdapde::core::advection;

// test 1
//    domain:       unit square [1,1] x [1,1] (coarse)
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
/*
TEST(kcv_srpde_test, laplacian_nonparametric_samplingatnodes_spaceonly_rmse) {
    // define domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/gcv/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 6, 1);
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);

    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, boundary_matrix);
    problem.set_forcing(u);
    problem.set_dirichlet_bc(DMatrix<double>::Zero(domain.mesh.n_nodes(), 1));

    // define model
    SRPDE model(problem, Sampling::mesh_nodes);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);

    model.init();    
    // define KCV engine and search for best lambda which minimizes the model's RMSE
    std::size_t n_folds = 4;
    KCV kcv(n_folds);

    std::vector<DVector<double>> lambdas;
    for (double x = -3.0; x <= 1.0; x += 0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
    kcv.fit(model, lambdas, RMSE(model));

    std::cout << kcv.avg_scores() << std::endl;

    std::cout << "----" << std::endl;

    std::cout << kcv.scores() << std::endl; 

    std::cout << "----" << std::endl;

    std::cout << kcv.optimum() << std::endl;

    // auto KCV_ = fdapde::calibration::KCV {n_folds}(lambdas, RMSE());
    // KCV_.fit(model);
    
    // test correctness
    // TODO
}
*/

TEST(kcv_srpde_test, transportTestCase1) {
    // define domain
    // MeshLoader<Mesh2D> domain("unit_square_coarse");

    constexpr std::size_t femOrder = 1;
    
    // import data from files
    // DMatrix<double> y = read_csv<double>("../data/models/gcv/2D_test1/y.csv");
    int n = 5; // range[10, 60]
    std::default_random_engine generator(123);
    DMatrix<double> locs = read_csv<double>("../data/transport/TransportTestCase1/locs" + std::to_string(n) + ".csv");
    DMatrix<double> observations = read_csv<double>("../data/transport/TransportTestCase1/observations" + std::to_string(n) + ".csv");
    std::normal_distribution<double> distribution(0.0, 0.05 * (observations.maxCoeff() - observations.minCoeff()));        
    DMatrix<double> y(n*n, 1);
    for (int j = 0; j < n*n; ++j)
        y(j, 0) = observations(j) + distribution(generator);  // generate noisy observations

    // define regularizing PDE
    // auto L = -laplacian<FEM>();
    // DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 6, 1);
    // DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);
    // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, boundary_matrix);
    // problem.set_forcing(u);
    // problem.set_dirichlet_bc(DMatrix<double>::Zero(domain.mesh.n_nodes(), 1));
    SVector<2> b;  b << 1., 1.;
    double nu = 1e-6;
    double BL = 0.1;
    double stabParam = 6.03; //23.294; //6.03;
    // define exact solution
    auto solutionExpr = [&BL](SVector<2> x) -> double {
        return x[0]*x[1]*x[1] - x[1]*x[1]*exp((2*(x[0] - 1))/BL) - x[0]*exp(3*(x[1] - 1)/BL) + exp((2*(x[0] - 1) + 3*(x[1] - 1))/BL);
    };
    // forcing term
    using std::exp;
    auto forcingExpr = [&nu, &b, &BL](SVector<2> x) -> double {
        return b[0]*(x[1]*x[1] - exp((3*x[1] - 3)/BL) - 2*x[1]*x[1]*exp((2*x[0] - 2)/BL)/BL + 2*exp((2*x[0] + 3*x[1] - 5)/BL)/BL) + b[1]*(2*x[0]*x[1] - 2*x[1]*exp((2*x[0] - 2)/BL) - 3*x[0]*exp((3*x[1] - 3)/BL)/BL + 3*exp((2*x[0] + 3*x[1] - 5)/BL)/BL) - nu*(2*x[0] - 2*exp((2*x[0] - 2)/BL) - 9*x[0]*exp((3*x[1] - 3)/BL)/(BL*BL) - 4*x[1]*x[1]*exp((2*x[0] - 2)/BL)/(BL*BL) + 13*exp((2*x[0] + 3*x[1] - 5)/BL)/(BL*BL));
    };
    ScalarField<2> forcing(forcingExpr);
    // define domain 
    MeshLoader<Mesh2D> domain("unit_square_64");
    //define regularizing PDE
    PDEparameters<decltype(nu), decltype(b)>::destroyInstance();
    PDEparameters<decltype(nu), decltype(b)> &PDEparams =
            PDEparameters<decltype(nu), decltype(b)>::getInstance(nu, b);
    auto L = - nu * laplacian<FEM>() + advection<FEM>(b);
    // define the boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);
    PDE< decltype(domain.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>, decltype(nu),
            decltype(b)> pde_(domain.mesh, L, boundary_matrix);
    pde_.set_forcing(forcing);
    pde_.set_stab_param(stabParam);
    pde_.set_dirichlet_bc(DMatrix<double>::Zero(domain.mesh.n_nodes(), 1));

    // define model
    SRPDE model(pde_, Sampling::pointwise);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);

    model.init();    
    // define KCV engine and search for best lambda which minimizes the model's RMSE
    std::size_t n_folds = 10;
    KCV kcv(n_folds);

    std::vector<DVector<double>> lambdas;
    for (double x = -6.0; x <= 3.0; x += 0.5) lambdas.push_back(SVector<1>(std::pow(10, x)));
    kcv.fit(model, lambdas, RMSE(model));

    std::cout << "kcv average scores:" << std::endl;
    std::cout << kcv.avg_scores() << std::endl;

    std::cout << "----" << std::endl;

    std::cout << "kcv scores matrix:" << std::endl;
    std::cout << kcv.scores() << std::endl; 

    std::cout << "----" << std::endl;

    std::cout << "kcv optimum lambda:" << std::endl;
    std::cout << kcv.optimum() << std::endl;

    for (std::size_t i = 0; i < lambdas.size(); ++i) {
        if (almost_equal(kcv.optimum()[0], lambdas[i][0], 1e-6)) {
            std::cout << "optimal lambda index: " << i << "/" << lambdas.size() << std::endl;
            break;
        }
    }

    std::cout << "----" << std::endl;

    std::cout << "model.f() size = " << model.f().rows() << "x" << model.f().cols() << std::endl;
}

/*
TEST(kcv_srpde_test, qsrpde_laplacian_nonparametric_samplingatnodes_spaceonly_rmse) {
    // define domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/qsrpde/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda = 1.778279 * std::pow(0.1, 4);
    double alpha = 0.1;
    QSRPDE<SpaceOnly> model(problem, Sampling::mesh_nodes, alpha);
    model.set_lambda_D(lambda);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    // define KCV engine and search for best lambda which minimizes the model's RMSE
    std::size_t n_folds = 5;
    KCV kcv(n_folds);
    std::vector<DVector<double>> lambdas;
    for (double x = -6.0; x <= -3.0; x += 0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
    kcv.fit(model, lambdas, RMSE(model));

    std::cout << kcv.avg_scores() << std::endl;
    
    // calibrator approach
    auto KCV_ = fdapde::calibration::KCV {n_folds}(lambdas, RMSE());
    KCV_.fit(model);
    
    // test correctness
    // TODO
}
*/