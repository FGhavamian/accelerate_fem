#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <map>
#include <string>

#include "model.h"


int main(int argc, char* argv[])
{
    (void) argc;

    std::map<std::string, std::string> path_config;
    std::map<std::string, double> material_config_1;
    std::map<std::string, double> material_config_2;
    std::map<int, std::map<std::string, double>> material_config;

    path_config["msh"] = argv[1];
    path_config["output_solution"] = argv[2];
    path_config["output_plastic_strain"] = argv[3];
    path_config["output_vm_stress"] = argv[4];

    material_config_1["b"] = std::stod(argv[5]);
    material_config_1["y"] = std::stod(argv[6]);

    material_config_2["b"] = std::stod(argv[7]);
    material_config_2["y"] = std::stod(argv[8]);

    int n_timestep = std::stoi(argv[9]);

    int verbose = std::stoi(argv[10]);

    material_config[1] = material_config_1;
    material_config[2] = material_config_2;

    Model model(material_config, path_config, n_timestep, verbose);
    model.run();
}


