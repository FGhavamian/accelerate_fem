#include <deal.II/base/symmetric_tensor.h>

#include "point_history.h"


PointHistory::PointHistory()
{
    stress_old = 0.0;
    stress = 0.0;
    strain_plastic_cum = 0.0;
    strain_plastic_cum_old = 0.0;
}

SymmetricTensor<2, 2> PointHistory::get_stress_old()
{
    return stress_old;
}

double PointHistory::get_strain_plastic_cum() const 
{
    return strain_plastic_cum_old;
}

void PointHistory::set_stress(SymmetricTensor<2, 2> stress_new)
{
    stress = stress_new;
}

void PointHistory::set_strain_plastic_cum(double strain_plastic_new)
{
    strain_plastic_cum = strain_plastic_new;
}

void PointHistory::update_history()
{
    stress_old = stress;
    strain_plastic_cum_old = strain_plastic_cum;
}