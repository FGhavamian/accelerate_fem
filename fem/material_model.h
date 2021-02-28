#ifndef MATERIAL_MODEL_H
#define MATERIAL_MODEL_H

#include <deal.II/base/symmetric_tensor.h>

#include "point_history.h"

using namespace dealii;


class MaterialModel
{
public:
    MaterialModel(std::map<std::string, double> config);

    void update(
            SymmetricTensor<2, 2>           strain_delta,
            std::shared_ptr<PointHistory>   &history,
            SymmetricTensor<2, 2>           &stress,
            SymmetricTensor<4, 2>           &stiffness);

private:
    double e;
    double nu;
    double n;
    double a;
    double b;
    double eta;
    double yield_stress;
    double dt;

    SymmetricTensor<4, 2> stiffness_elastic;

    SymmetricTensor<4, 2> get_elastic_stiffness();
    double update_yield_stress(double k);
    double compute_von_mises_stress(SymmetricTensor<2, 2> stress_);
    double compute_yeild_func_val(double stress_von_mises, double stress_yield_new);

    SymmetricTensor<2, 2> compute_direction_of_plastic_strain(
            SymmetricTensor<2, 2> stress_new,
            double stress_von_mises);

    SymmetricTensor<4, 2> compute_derivative_direction_of_plastic_strain_stress(
            SymmetricTensor<2, 2> direction_of_plastic_strain,
            double stress_von_mises);

    double compute_derivative_yield_func_plastic_strain(double strain_plastic_new);
    double compute_over_stress_coef(double yield_func_val, double stress_yield_new);
    double compute_derivative_over_stress_coef_yield_func_val(double yield_func_val, double stress_yield_new);
};

#endif