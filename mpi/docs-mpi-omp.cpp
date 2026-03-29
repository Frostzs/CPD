#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

// ----------------------------------------------------------------------
//
//  Collaborators: Eduardo Barata, Fábio Prata, Carlos Alexandre
//
// ----------------------------------------------------------------------


int cabinets, documents, numSubjects;

// calculates the distances between a document in the local buffer and a centroid
double computeDistance(int local_doc_idx, int cab_idx, const std::vector<double>& local_scores, const std::vector<double>& centroids) {
    double dist = 0.0;
    #pragma omp simd reduction(+:dist)
    for (int s = 0; s < numSubjects; s++) {
        double diff = local_scores[local_doc_idx * numSubjects + s] - centroids[cab_idx * numSubjects + s];
        dist += diff * diff;
    }
    return dist;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    double exec_time;
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
    if (rank == 0)
    {
        exec_time = -omp_get_wtime();
    }

    // share metadata
    MPI_Bcast(&cabinets, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&documents, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSubjects, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // split the work
    std::vector<int> n_docs(size), range_docs(size);
    std::vector<double> centroids(cabinets * numSubjects);
    int base = documents / size;
    int mod  = documents % size;

    for (int i = 0; i < size; i++) {
        n_docs[i] = base + (i < mod ? 1 : 0);   // add the ramainder from the beggining
        range_docs[i] = (i == 0) ? 0 : range_docs[i - 1] + n_docs[i - 1];   // the range each thread is gonna work on (ex: [0..n], [n+1.. n+n], etc)
    }

    int local_n = n_docs[rank];

    // we have to multiply the values with numSubjects because of initial reading from file
    std::vector<int> doc_scores(size), range_scores(size);
    for (int i = 0; i < size; i++) {
        doc_scores[i] = n_docs[i] * numSubjects;
        range_scores[i] = range_docs[i] * numSubjects;
    }

    std::vector<double> local_scores(local_n * numSubjects);

    // scatter the data
    MPI_Scatterv(
        rank == 0 ? all_scores_flat.data() : nullptr,   // sendbuf
        doc_scores.data(),                           // sendcounts
        range_scores.data(),                           // displs
        MPI_DOUBLE,                                     //sendtype
        local_scores.data(),                            // recvbuf
        doc_scores[rank],                            // recvcount
        MPI_DOUBLE,                                     // recvtype
        0,                                              // root
        MPI_COMM_WORLD                                  // comm
    );


    std::vector<int> local_assignment(local_n);
    
    // start parallel region
    #pragma omp parallel
    {
        // initial round-robin assignment
        #pragma omp for simd schedule(static)
        for (int i = 0; i < local_n; i++) {
            int global_i = range_docs[rank] + i;
            local_assignment[i] = global_i % cabinets;
        }


        bool changed = true;
        while (changed) {
            std::vector<double> local_sums(cabinets * numSubjects, 0.0);
            std::vector<int> local_counts(cabinets, 0);

            // sum up scores for assigned cabinets
            #pragma omp for schedule(static)
            for (int i = 0; i < local_n; i++) {
                int c = local_assignment[i];
                local_counts[c]++;
                for (int s = 0; s < numSubjects; s++)
                    local_sums[c * numSubjects + s] += local_scores[i * numSubjects + s];
            }

            // sync sums
            std::vector<double> global_sums(cabinets * numSubjects);
            std::vector<int> global_counts(cabinets);
            #pragma omp master
            {
                MPI_Allreduce(local_sums.data(), global_sums.data(), cabinets * numSubjects, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(local_counts.data(), global_counts.data(), cabinets, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
            #pragma omp barrier
            
            // update Centroids (means)
            #pragma omp for schedule(static)
            for (int c = 0; c < cabinets; c++) {
                if (global_counts[c] > 0) {
                    for (int s = 0; s < numSubjects; s++)
                        centroids[c * numSubjects + s] = global_sums[c * numSubjects + s] / global_counts[c];
                }
            }

            // find the new closest cabinet
            bool local_changed = false;
            #pragma omp for schedule(static)
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
                    #pragma omp atomic write
                    local_changed = true;
                }
            }

            #pragma omp master
            {
                // stop if NO ONE changed an assignment
                MPI_Allreduce(&local_changed, &changed, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            }
            #pragma omp barrier
        }
    }
    

    std::vector<int> final_assignments;
    if (rank == 0) final_assignments.resize(documents);

    MPI_Gatherv(
        local_assignment.data(),
        local_n,
        MPI_INT,
        rank == 0 ? final_assignments.data() : nullptr,
        n_docs.data(),
        range_docs.data(),
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        exec_time += omp_get_wtime();
        fprintf(stderr, "%.1fs\n", exec_time);
        for (int i = 0; i < documents; i++) std::cout << final_assignments[i] << "\n";
    }

    MPI_Finalize();
    return 0;
}