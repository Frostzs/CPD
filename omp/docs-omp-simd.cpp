#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>

int main(int argc, char const *argv[]) {
    if (argc < 2) return 1;

    int cabinets, documents, numSubjects;
    std::ifstream file(argv[1]);
    file >> cabinets >> documents >> numSubjects;

    std::vector<double> scores(documents * numSubjects);
    for (int i = 0; i < documents; i++) {
        int id; file >> id;
        for (int s = 0; s < numSubjects; s++) 
            file >> scores[i * numSubjects + s];
    }
    file.close();

    std::vector<double> means(cabinets * numSubjects, 0.0);
    std::vector<int> assignments(documents);
    for (int i = 0; i < documents; i++) assignments[i] = i % cabinets;

    double exec_time = -omp_get_wtime();
    bool changed = true;

    while (changed) {
        changed = false;

        // 1. Reset and Update Means (Parallel)
        std::vector<int> counts(cabinets, 0);
        std::fill(means.begin(), means.end(), 0.0);

        #pragma omp parallel
        {
            std::vector<double> local_means(cabinets * numSubjects, 0.0);
            std::vector<int> local_counts(cabinets, 0);

            #pragma omp for nowait
            for (int i = 0; i < documents; i++) {
                int c = assignments[i];
                local_counts[c]++;

                for (int s = 0; s < numSubjects; s++) {
                    local_means[c * numSubjects + s] += scores[i * numSubjects + s];
                }
            }

            #pragma omp critical
            {
                for (int c = 0; c < cabinets; c++) {
                    counts[c] += local_counts[c];
                    for (int s = 0; s < numSubjects; s++)
                        means[c * numSubjects + s] += local_means[c * numSubjects + s];
                }
            }
        }

        for (int c = 0; c < cabinets; c++) {
            if (counts[c] > 0) {
                for (int s = 0; s < numSubjects; s++) 
                    means[c * numSubjects + s] /= counts[c];
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < documents; i++) {
            int best_cabinet = 0;
            double min_dist = INFINITY;

            for (int c = 0; c < cabinets; c++) {
                double dist = 0.0;
                
                #pragma omp simd reduction(+:dist)
                for (int s = 0; s < numSubjects; s++) {
                    double diff = scores[i * numSubjects + s] - means[c * numSubjects + s];
                    dist += diff * diff;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cabinet = c;
                }
            }

            if (assignments[i] != best_cabinet) {
                assignments[i] = best_cabinet;
                #pragma omp atomic write
                changed = true;
            }
        }
    }

    exec_time += omp_get_wtime();
    fprintf(stderr, "%.1fs\n", exec_time);
    
    for (int i = 0; i < documents; i++) 
        std::cout << assignments[i] << "\n";

    return 0;
}
