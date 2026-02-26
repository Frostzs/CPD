#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <vector>

int cabinets, documents, numSubjects = 0;
double delta = 0.0;

// information of a document
struct docs
{
    int id;
    std::vector<double> score;
    std::vector<double> distances;
};

struct container 
{ 
    int id;
    std::vector<double> mean;
    std::vector<docs> docs_vector;
};

void updateMean (container &c){
    double total_value = 0.0;
    for (size_t i = 0; i < numSubjects; i++){
        total_value = 0.0;
        for (size_t j = 0; j < c.docs_vector.size(); j++){
            total_value += c.docs_vector[j].score[i];
        }
        c.mean[i] = total_value / c.docs_vector.size();
    }
}

void computeDistance (container &c, docs &d){
    double sum = 0.0;
    for (size_t i = 0; i < numSubjects; i++){
        delta += (d.score[i] - c.mean[i]) * (d.score[i] - c.mean[i]);
    }
    d.distances[c.id] = delta;
}

// read a file
void readFile (char const* filename, std::vector<docs>& info)
{

    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        exit(1);
    }

    // gets the first line and puts it into the respective variables
    file >> cabinets >> documents >> numSubjects;

    // gets every id and score a puts it into the docs struct
    for (size_t i = 0; i < documents; i++)
    {
        docs d;
        file >> d.id;

        d.score.resize(numSubjects);
        for (int s = 0; s < numSubjects; s++) {
            file >> d.score[s];
        }

        info.push_back(d);
    }
    
    // can't forget to close file
    file.close();
    return;
}


int main(int argc, char const *argv[])
{
    std::vector<docs> info;

    if (argc < 2) {
        std::cerr << "Usage: ./docs <Name of the file>\n";
        return 1;
    }

    readFile(argv[1], info);
    

    return 0;
}
