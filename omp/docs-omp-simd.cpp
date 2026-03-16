#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h> 

int cabinets, documents, numSubjects = 0;

// information of a document
struct docs
{
    int id;
    std::vector<double> score;
    std::vector<double> distances;
};

// container so we dont confuse with other variables but its supposed to be cabinet
struct container
{
    int id;
    std::vector<double> mean;
    std::vector<docs> docs_vector;
};

// updates the mean inside of a cabinet
void updateMean(container &c)
{
    if (c.docs_vector.empty()) {
        std::fill(c.mean.begin(), c.mean.end(), 0.0);
        return;
    }

    for (int s = 0; s < numSubjects; s++) {
        double total = 0.0;
        
        
        for (size_t j = 0; j < c.docs_vector.size(); j++)
            total += c.docs_vector[j].score[s];

        c.mean[s] = total / c.docs_vector.size();
    }
}

// compute the distance of a document to a cabinet
void computeDistance(container &c, docs &d)
{
    double sum = 0.0;

    // Only suitable spot for SIMD
    #pragma omp simd
    for (int s = 0; s < numSubjects; s++) {
        double diff = d.score[s] - c.mean[s];
        sum += diff * diff;
    }

    d.distances[c.id] = sum;
}

// read a file
void readFile(const char* filename, std::vector<docs>& info)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file\n";
        exit(1);
    }

    file >> cabinets >> documents >> numSubjects;

    info.reserve(documents);

    for (int i = 0; i < documents; i++)
    {
        docs d;
        file >> d.id;

        d.score.resize(numSubjects);
        d.distances.resize(cabinets);

        for (int s = 0; s < numSubjects; s++)
            file >> d.score[s];

        info.push_back(d);
    }

    file.close();
}


int main(int argc, char const *argv[])
{
    double exec_time;

    if (argc < 2) {
        std::cerr << "Usage: ./docs <file>\n";
        return 1;
    }

    std::vector<docs> info;
    readFile(argv[1], info);

    exec_time = -omp_get_wtime();

    // track which cabinet each document belongs to
    std::vector<int> assignment(documents);

    // initialize cabinets
    std::vector<container> cabs(cabinets);
    
    for (int c = 0; c < cabinets; c++) {
        cabs[c].id = c;
        cabs[c].mean.resize(numSubjects);
        cabs[c].docs_vector.reserve(documents / cabinets + 1);
    }

    // round-robin initial assignment
    for (int i = 0; i < documents; i++) {
        assignment[i] = i % cabinets;
    }

    bool changed = true;

    while (changed)
    {
        changed = false;

        // update means for each cabinet and rebuilt its vector
        #pragma omp parallel
        {
            // schedule(static): divides workload uniformly between threads (avoids runtime scheduling)
            #pragma omp for schedule(static)
            for (int c = 0; c < cabinets; c++) {
                cabs[c].docs_vector.clear();
                for (int i = 0; i < documents; i++) {
                    if (assignment[i] == c) {
                        cabs[c].docs_vector.push_back(info[i]);
                    }
                }
                updateMean(cabs[c]);
            }
            

            #pragma omp for schedule(static)
            // assign documents to closest cabinet
            for (int i = 0; i < documents; i++)
            {
                docs &d = info[i];

                // compute the distances to all cabinets
                //probably pragmaa omp for here
                for (int c = 0; c < cabinets; c++)
                    computeDistance(cabs[c], d);


                int best = 0;
                double best_dist = d.distances[0];

                // searches for the lowest distance of a document to a cabinet
                //pragmaa omp for here also
                for (int c = 1; c < cabinets; c++) {
                    if (d.distances[c] < best_dist) {
                        best_dist = d.distances[c];
                        best = c;
                    }
                }

                // if found a new distance, change it in assignment
                if (best != assignment[i]) {
                    assignment[i] = best;
                    #pragma omp atomic write
                    changed = true;
                }
            }
        }
    }

    exec_time += omp_get_wtime();
    fprintf(stderr, "%.1fs\n", exec_time);

    // output ordered by document id
    for (int i = 0; i < documents; i++)
        std::cout << assignment[info[i].id] << "\n";

    return 0;
}