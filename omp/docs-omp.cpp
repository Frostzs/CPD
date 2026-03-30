#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>

int main(int argc, char const *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./docs <file>\n";
        return 1;
    }

    int cabinets, documents, numSubjects;
    
    // --- 1. Flattened File Reading ---
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file\n";
        return 1;
    }
    file >> cabinets >> documents >> numSubjects;

    std::vector<std::vector<double>> scores(documents, std::vector<double>(numSubjects));
    for (int i = 0; i < documents; i++) {
        int id; 
        file >> id; 
        for (int s = 0; s < numSubjects; s++) file >> scores[i][s];
    }
    file.close();

    // --- 2. State Initialization ---
    std::vector<std::vector<double>> means(cabinets, std::vector<double>(numSubjects, 0.0));
    std::vector<int> assignments(documents);
    for (int i = 0; i < documents; i++) assignments[i] = i % cabinets;

    double exec_time = -omp_get_wtime();
    bool changed = true;

    while (changed) {
        changed = false;

        // --- 3. Parallel Mean Update ---
        // We reset means and counts for the new iteration
        std::vector<int> counts(cabinets, 0);
        for (int c = 0; c < cabinets; c++) {
            std::fill(means[c].begin(), means[c].end(), 0.0);
        }

        #pragma omp parallel
        {
            // Thread-local accumulation to avoid massive contention
            std::vector<std::vector<double>> local_means(cabinets, std::vector<double>(numSubjects, 0.0));
            std::vector<int> local_counts(cabinets, 0);

            #pragma omp for nowait
            for (int i = 0; i < documents; i++) {
                int c = assignments[i];
                local_counts[c]++;
                for (int s = 0; s < numSubjects; s++) {
                    local_means[c][s] += scores[i][s];
                }
            }

            #pragma omp critical
            {
                for (int c = 0; c < cabinets; c++) {
                    counts[c] += local_counts[c];
                    for (int s = 0; s < numSubjects; s++) {
                        means[c][s] += local_means[c][s];
                    }
                }
            }
        }

        for (int c = 0; c < cabinets; c++) {
            if (counts[c] > 0) {
                for (int s = 0; s < numSubjects; s++) means[c][s] /= counts[c];
            }
        }

        // --- 4. Parallel Reassignment ---
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < documents; i++) {
            int best_cabinet = 0;
            double min_dist = INFINITY;

            for (int c = 0; c < cabinets; c++) {
                double dist = 0.0;
                for (int s = 0; s < numSubjects; s++) {
                    double diff = scores[i][s] - means[c][s];
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

    // --- 5. Output ---
    for (int i = 0; i < documents; i++) std::cout << assignments[i] << "\n";

    return 0;
}
