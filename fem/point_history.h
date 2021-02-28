#ifndef POINT_HISTORY_H
#define POINT_HISTORY_H

#include <deal.II/base/symmetric_tensor.h>


using namespace dealii;


class PointHistory
{
public:
    PointHistory();

    SymmetricTensor<2, 2>   get_stress_old();
    double                  get_strain_plastic_cum() const;

    void                    set_stress(SymmetricTensor<2, 2> stress_new);
    void                    set_strain_plastic_cum(double strain_plastic_new);

    void update_history();

private:
    SymmetricTensor<2, 2> stress;
    SymmetricTensor<2, 2> stress_old;
    double                strain_plastic_cum;
    double                strain_plastic_cum_old;
};

#endif