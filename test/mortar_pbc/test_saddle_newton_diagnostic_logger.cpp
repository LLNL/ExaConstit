// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors
//
// Unit tests for SaddleNewtonDiagnosticLogger's Newton/linear solve CSV path.

#include "mortar_pbc/saddle_newton_diagnostic_logger.hpp"
#include "mortar_pbc/saddle_residual_scaler.hpp"

#include "mfem.hpp"
#include "mpi.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{

void AssertOrDie(bool cond, const std::string& test_name,
                 const std::string& detail)
{
    if (!cond)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail << std::endl;
        std::exit(1);
    }
}

void AssertNear(double a, double b, double tol,
                const std::string& test_name,
                const std::string& detail)
{
    if (std::abs(a - b) > tol)
    {
        std::cerr << "  FAIL  " << test_name << ": " << detail
                  << " (got " << a << ", expected " << b << ")"
                  << std::endl;
        std::exit(1);
    }
}

std::vector<std::string> SplitCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ','))
    {
        fields.push_back(field);
    }
    return fields;
}

std::shared_ptr<mortar_pbc::SaddleResidualScaler> MakeScaler()
{
    mortar_pbc::SaddleResidualScalerConfig cfg;
    cfg.enabled = true;
    cfg.per_subblock = true;

    auto scaler = std::make_shared<mortar_pbc::SaddleResidualScaler>(cfg);

    mfem::Array<int> subblock_of_row(3);
    subblock_of_row[0] = 0;
    subblock_of_row[1] = 1;
    subblock_of_row[2] = 1;
    scaler->SetPartitionDirect({"edge", "face"}, subblock_of_row);
    return scaler;
}

void TestLoggerWritesLinearSolveColumns()
{
    const std::string name =
        "SaddleNewtonDiagnosticLogger linear solve columns";
    const std::string filename =
        "/tmp/exaconstit_test_saddle_newton_diagnostic_logger.csv";
    std::remove(filename.c_str());

    mfem::Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = 2;
    offsets[2] = 5;

    {
        auto scaler = MakeScaler();
        mortar_pbc::SaddleNewtonDiagnosticLogger logger(
            scaler, offsets, MPI_COMM_WORLD, filename);

        NewtonDiagnosticSink newton_sink = logger.MakeSink();
        LinearSolveDiagnosticSink linear_sink =
            logger.MakeLinearSolveSink();

        mfem::Vector residual(5);
        residual[0] = 3.0;
        residual[1] = 4.0;
        residual[2] = 1.0;
        residual[3] = 2.0;
        residual[4] = 2.0;

        NewtonIterDiagnostic iter0;
        iter0.iter = 0;
        iter0.norm = 6.0;
        iter0.norm0 = 6.0;
        iter0.norm_max = 1.0e-8;
        iter0.converged_now = false;
        iter0.residual = &residual;
        newton_sink(iter0);

        LinearSolveDiagnostic lin0;
        lin0.iterations = 7;
        lin0.final_norm = 2.5e-9;
        lin0.converged = true;
        linear_sink(lin0);

        NewtonIterDiagnostic iter1 = iter0;
        iter1.iter = 1;
        iter1.norm = 0.0;
        iter1.converged_now = true;
        newton_sink(iter1);
        logger.IncrementStep();
    }

    std::ifstream in(filename);
    AssertOrDie(in.is_open(), name, "failed to open diagnostic CSV");

    std::string header_line;
    std::string row0_line;
    std::string row1_line;
    std::getline(in, header_line);
    std::getline(in, row0_line);
    std::getline(in, row1_line);

    const std::vector<std::string> header = SplitCsvLine(header_line);
    const std::vector<std::string> row0 = SplitCsvLine(row0_line);
    const std::vector<std::string> row1 = SplitCsvLine(row1_line);

    const std::vector<std::string> expected_header = {
        "step", "iter", "norm", "norm0", "norm_max", "converged_now",
        "linear_iterations", "linear_final_norm", "linear_converged",
        "scaler_enabled", "res_K", "res_lam", "res_lam_edge",
        "res_lam_face", "d_u", "d_lam_edge", "d_lam_face"};
    AssertOrDie(header == expected_header, name,
                "unexpected CSV header");
    AssertOrDie(row0.size() == expected_header.size(), name,
                "unexpected first row field count");
    AssertOrDie(row1.size() == expected_header.size(), name,
                "unexpected second row field count");

    AssertOrDie(std::stoi(row0[0]) == 0 && std::stoi(row0[1]) == 0,
                name, "first row step/iter mismatch");
    AssertOrDie(std::stoi(row0[6]) == 7, name,
                "linear iteration count was not written");
    AssertNear(std::stod(row0[7]), 2.5e-9, 1.0e-20, name,
               "linear final norm was not written");
    AssertOrDie(std::stoi(row0[8]) == 1, name,
                "linear convergence flag was not written");
    AssertNear(std::stod(row0[10]), 5.0, 1.0e-14, name,
               "K-block residual norm mismatch");
    AssertNear(std::stod(row0[11]), 3.0, 1.0e-14, name,
               "lambda-block residual norm mismatch");

    AssertOrDie(std::stoi(row1[0]) == 0 && std::stoi(row1[1]) == 1,
                name, "second row step/iter mismatch");
    AssertOrDie(std::stoi(row1[6]) == -1, name,
                "converged Newton row should record no linear solve");
    AssertNear(std::stod(row1[7]), -1.0, 0.0, name,
               "converged Newton row should record no final norm");
    AssertOrDie(std::stoi(row1[8]) == 0, name,
                "converged Newton row should record false linear convergence");

    std::remove(filename.c_str());
    std::cout << "  PASS  " << name << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "Running SaddleNewtonDiagnosticLogger tests"
                  << std::endl;
        std::cout << "------------------------------------------------"
                  << std::endl;
    }

    TestLoggerWritesLinearSolveColumns();

    if (rank == 0)
    {
        std::cout << "------------------------------------------------"
                  << std::endl;
        std::cout << "All SaddleNewtonDiagnosticLogger tests passed."
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
