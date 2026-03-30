#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>

// ----------------------------------------------------------------------
//
//  Collaborators: Eduardo Barata, Fábio Prata, Carlos Alexandre
//
// ----------------------------------------------------------------------

int main(int argc, char const *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./docs <file>\n";
        return 1;
    }

    // Global variables replaced by locals
    int cabinets, documents, numSubjects;
    
    // --- FILE READING (Flattened) ---
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file\n";
        return 1;
    }

    file >> cabinets >> documents >> numSubjects;

    // We use a 2D vector for scores: scores[doc_id][subject]
    std::vector<std::vector<double>> scores(documents, std::vector<double>(numSubjects));
    for (int i = 0; i < documents; i++) {
        int id; 
        file >> id; // Assuming IDs are 0 to documents-1
        for (int s = 0; s < numSubjects; s++) {
            file >> scores[id][s];
        }
    }
    file.close();

    double exec_time = -omp_get_wtime();

    // --- INITIALIZATION ---
    // Instead of container struct, we use separate vectors
    std::vector<std::vector<double>> means(cabinets, std::vector<double>(numSubjects, 0.0));
    std::vector<int> assignments(documents);

    // Initial round-robin assignment
    for (int i = 0; i < documents; i++) {
        assignments[i] = i % cabinets;
    }

    bool changed = true;
    while (changed) {
        changed = false;

        // Update Means
        // Reset means to zero
        for (int c = 0; c < cabinets; c++) {
            std::fill(means[c].begin(), means[c].end(), 0.0);
        }

        std::vector<int> counts(cabinets, 0);
        for (int i = 0; i < documents; i++) {
            int c = assignments[i];
            counts[c]++;
            for (int s = 0; s < numSubjects; s++) {
                means[c][s] += scores[i][s];
            }
        }

        for (int c = 0; c < cabinets; c++) {
            if (counts[c] > 0) {
                for (int s = 0; s < numSubjects; s++) {
                    means[c][s] /= counts[c];
                }
            }
        }

        // Reassign Documents
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
                changed = true;
            }
        }
    }

    exec_time += omp_get_wtime();
    fprintf(stderr, "%.1fs\n", exec_time);

    // --- OUTPUT ---
    for (int i = 0; i < documents; i++) {
        std::cout << assignments[i] << "\n";
    }

    return 0;
}
