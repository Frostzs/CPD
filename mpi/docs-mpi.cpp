#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

int cabinets, documents, numSubjects;

// Helper: Calculate distance between a document in the local buffer and a centroid
double computeDistance(int local_doc_idx, int cab_idx, const std::vector<double>& local_scores, const std::vector<double>& centroids) {
    double dist = 0.0;
    for (int s = 0; s < numSubjects; s++) {
        double diff = local_scores[local_doc_idx * numSubjects + s] - centroids[cab_idx * numSubjects + s];
        dist += diff * diff;
    }
    return dist;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> all_scores_flat;
    if (rank == 0) {
        std::ifstream file(argv[1]);
        if (!file.is_open()) { MPI_Abort(MPI_COMM_WORLD, 1); }
        file >> cabinets >> documents >> numSubjects;
        all_scores_flat.resize(documents * numSubjects);
        int dummy_id;
        for (int i = 0; i < documents; i++) {
            file >> dummy_id;
            for (int s = 0; s < numSubjects; s++) 
                file >> all_scores_flat[i * numSubjects + s];
        }
        file.close();
    }

    // Share metadata
    MPI_Bcast(&cabinets, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&documents, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSubjects, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Split the work
    int local_n = documents / size; 
    std::vector<double> local_scores(local_n * numSubjects);
    
    // Scatter data: Each process gets its own slice of documents
    MPI_Scatter(all_scores_flat.data(), local_n * numSubjects, MPI_DOUBLE, 
                local_scores.data(), local_n * numSubjects, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> centroids(cabinets * numSubjects);
    std::vector<int> local_assignment(local_n);
    
    // Initial round-robin assignment
    for (int i = 0; i < local_n; i++) local_assignment[i] = (rank * local_n + i) % cabinets;

    bool changed = true;
    while (changed) {
        std::vector<double> local_sums(cabinets * numSubjects, 0.0);
        std::vector<int> local_counts(cabinets, 0);

        // 1. Local Math: Sum up scores for assigned cabinets
        for (int i = 0; i < local_n; i++) {
            int c = local_assignment[i];
            local_counts[c]++;
            for (int s = 0; s < numSubjects; s++)
                local_sums[c * numSubjects + s] += local_scores[i * numSubjects + s];
        }

        // 2. Global Sync: Combine everyone's sums and counts
        std::vector<double> global_sums(cabinets * numSubjects);
        std::vector<int> global_counts(cabinets);
        MPI_Allreduce(local_sums.data(), global_sums.data(), cabinets * numSubjects, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(), cabinets, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // 3. Update Centroids (Means)
        for (int c = 0; c < cabinets; c++) {
            if (global_counts[c] > 0) {
                for (int s = 0; s < numSubjects; s++)
                    centroids[c * numSubjects + s] = global_sums[c * numSubjects + s] / global_counts[c];
            }
        }

        // 4. Re-assign: Find the new closest cabinet
        bool local_changed = false;
        for (int i = 0; i < local_n; i++) {
            int best_c = 0;
            double min_dist = 1e30;
            for (int c = 0; c < cabinets; c++) {
                double d = computeDistance(i, c, local_scores, centroids);
                if (d < min_dist) {
                    min_dist = d;
                    best_c = c;
                }
            }
            if (local_assignment[i] != best_c) {
                local_assignment[i] = best_c;
                local_changed = true;
            }
        }

        // 5. Global Check: Stop if NO ONE changed an assignment
        MPI_Allreduce(&local_changed, &changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    }

    // Gather final results back to Rank 0
    std::vector<int> final_assignments;
    if (rank == 0) final_assignments.resize(documents);
    MPI_Gather(local_assignment.data(), local_n, MPI_INT, 
               final_assignments.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < documents; i++) std::cout << final_assignments[i] << "\n";
    }

    MPI_Finalize();
    return 0;
}