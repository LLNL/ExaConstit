// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) ExaConstit contributors

#include "mortar_pbc/amgf_utils.hpp"

#include "utilities/mechanics_log.hpp"

#include <algorithm>
#include <numeric>

namespace exaconstit::amgf {

namespace {

std::vector<HYPRE_BigInt> GatherUniqueIndices(
    const std::vector<HYPRE_BigInt>& local_idx,
    HYPRE_BigInt n_global_rows,
    MPI_Comm comm)
{
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);

    for (HYPRE_BigInt idx : local_idx)
    {
        MFEM_VERIFY(idx >= 0 && idx < n_global_rows,
                    "BuildBooleanRestrictionProlongation: index "
                    << idx << " outside valid range [0, "
                    << n_global_rows << ")");
    }

    const int n_local = static_cast<int>(local_idx.size());
    std::vector<int> counts(nranks, 0);
    MPI_Allgather(&n_local, 1, MPI_INT,
                  counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(nranks, 0);
    int n_total = 0;
    for (int r = 0; r < nranks; ++r)
    {
        displs[r] = n_total;
        n_total += counts[r];
    }

    std::vector<HYPRE_BigInt> all_idx(static_cast<std::size_t>(n_total));
    MPI_Allgatherv(local_idx.data(), n_local, HYPRE_MPI_BIG_INT,
                   all_idx.data(), counts.data(), displs.data(),
                   HYPRE_MPI_BIG_INT, comm);

    std::sort(all_idx.begin(), all_idx.end());
    all_idx.erase(std::unique(all_idx.begin(), all_idx.end()),
                  all_idx.end());
    return all_idx;
}

}  // namespace

mfem::HypreParMatrix* BuildBooleanRestrictionProlongation(
    HYPRE_BigInt n_global_rows,
    const std::vector<HYPRE_BigInt>& idx_global,
    const HYPRE_BigInt* k_row_starts,
    MPI_Comm comm)
{
    CALI_CXX_MARK_SCOPE(
        "exaconstit::amgf::build_boolean_restriction_prolongation");

    MFEM_VERIFY(k_row_starts != nullptr,
                "BuildBooleanRestrictionProlongation: k_row_starts is null");
    MFEM_VERIFY(k_row_starts[0] <= k_row_starts[1],
                "BuildBooleanRestrictionProlongation: invalid local K row "
                "partition [" << k_row_starts[0] << ", "
                << k_row_starts[1] << ")");

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    const std::vector<HYPRE_BigInt> idx_union =
        GatherUniqueIndices(idx_global, n_global_rows, comm);

    const HYPRE_BigInt row_first = k_row_starts[0];
    const HYPRE_BigInt row_end = k_row_starts[1];
    const int n_local_rows = static_cast<int>(row_end - row_first);

    int n_local_cols = 0;
    for (HYPRE_BigInt idx : idx_union)
    {
        if (idx >= row_first && idx < row_end)
        {
            ++n_local_cols;
        }
    }

    std::vector<int> all_local_cols(nranks, 0);
    MPI_Allgather(&n_local_cols, 1, MPI_INT,
                  all_local_cols.data(), 1, MPI_INT, comm);

    HYPRE_BigInt col_first = 0;
    for (int r = 0; r < rank; ++r)
    {
        col_first += all_local_cols[r];
    }
    const HYPRE_BigInt n_global_cols =
        std::accumulate(all_local_cols.begin(), all_local_cols.end(),
                        HYPRE_BigInt{0});

    std::vector<HYPRE_BigInt> col_starts(2);
    col_starts[0] = col_first;
    col_starts[1] = col_first + n_local_cols;

    mfem::SparseMatrix local_block(
        n_local_rows, static_cast<int>(n_global_cols));

    HYPRE_BigInt local_col = col_first;
    for (HYPRE_BigInt idx : idx_union)
    {
        if (idx < row_first || idx >= row_end)
        {
            continue;
        }
        const int local_row = static_cast<int>(idx - row_first);
        local_block.Add(local_row, static_cast<int>(local_col), 1.0);
        ++local_col;
    }
    local_block.Finalize();

    auto* P = new mfem::HypreParMatrix(
        comm,
        n_local_rows,
        n_global_rows,
        n_global_cols,
        local_block.ReadI(false),
        local_block.ReadJ(false),
        local_block.ReadData(false),
        k_row_starts,
        col_starts.data());

    return P;
}

}  // namespace exaconstit::amgf
