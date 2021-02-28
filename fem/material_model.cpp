#include "material_model.h"
#include "point_history.h"


double delta_func(unsigned int i, unsigned int j)
{
    return (i==j) ? 1.0 : 0.0;
}

MaterialModel::MaterialModel(std::map<std::string, double> config)
{
    e = 1000.0;
    nu = 0.0;
    n = 1.0;
    a = -1.0;
    b = config["b"];
    eta = 1e-5;
    yield_stress = config["y"];
    dt = 75;

    stiffness_elastic = get_elastic_stiffness();
}

void MaterialModel::update(
        SymmetricTensor<2, 2>          strain_delta,
        std::shared_ptr<PointHistory>  &history,
        SymmetricTensor<2, 2>          &stress,
        SymmetricTensor<4, 2>          &stiffness)
{
    SymmetricTensor<2, 2> stress_old         = history->get_stress_old();
    double                strain_plastic_cum_old = history->get_strain_plastic_cum();

    stress = stress_old + stiffness_elastic * strain_delta;
    stiffness = stiffness_elastic;

    double strain_plastic_new = strain_plastic_cum_old;

    double yield_stress_new = update_yield_stress(strain_plastic_new);
    double stress_von_mises = compute_von_mises_stress(stress);
    double yield_func_val = compute_yeild_func_val(stress_von_mises, yield_stress_new);

    if (yield_func_val > 0)
    {
        int counter = 0;
        int max_count = 10;
        double tol = 1e-12;
        bool converged = false;

        double k = strain_plastic_new;

        while (!converged)
        {
            counter++;

            double y_ = update_yield_stress(k);
            double s_vm = compute_von_mises_stress(stress);
            double f = compute_yeild_func_val(s_vm, y_);

            double df_dk = compute_derivative_yield_func_plastic_strain(k);

            SymmetricTensor<2, 2> m = compute_direction_of_plastic_strain(stress, s_vm);
            SymmetricTensor<4, 2> dm_ds = compute_derivative_direction_of_plastic_strain_stress(m, s_vm);

            double phi = compute_over_stress_coef(f, y_);
            double dphi_df = compute_derivative_over_stress_coef_yield_func_val(f, y_);

            SymmetricTensor<4, 2> p = identity_tensor<2>() + k * stiffness_elastic * dm_ds;
            SymmetricTensor<4, 2> p_inv = invert(p);

            SymmetricTensor<2, 2> r_s = stress-stress_old - stiffness_elastic*strain_delta + k*stiffness_elastic*m;
            double r_k                = k - eta * dt * phi;

            double dk = - m*p_inv*r_s;
            dk -= r_k/(eta * dt * dphi_df);
            dk /= m*p_inv*stiffness_elastic*m + df_dk + 1/(eta * dt * dphi_df);
            SymmetricTensor<2, 2> ds = -p_inv*r_s - p_inv*stiffness_elastic*m*dk;

            k += dk;
            stress += ds;

            double res = r_s.norm() + abs(r_k);
            // std::cout << "counter: " << counter << ", res: " << res << std::endl;
            if (res<tol || counter==max_count)
            {
                converged = true;

                SymmetricTensor<4, 2> r = p_inv*stiffness_elastic;
                double a_ = m*r*m;
                a_ += df_dk + 1/(eta * dt * dphi_df);

                stiffness = r - outer_product(r*m, r*m)/a_;
                strain_plastic_new = k;
            }
        }
    }

    history->set_stress(stress);
    history->set_strain_plastic_cum(strain_plastic_new);
}


SymmetricTensor<4, 2> MaterialModel::get_elastic_stiffness()
{
    SymmetricTensor<4, 2> tmp;
    for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = 0; j < 2; ++j)
            for (unsigned int k = 0; k < 2; ++k)
                for (unsigned int l = 0; l < 2; ++l)
                    tmp[i][j][k][l] = e/(2*(1+nu)) *
                                            (delta_func(i,l)*delta_func(j,k) + delta_func(i,k)*delta_func(j,l))
                                    + e*nu/((1+nu) * (1-2*nu)) * (delta_func(i,j) * delta_func(k,l));

    return tmp;
}

double MaterialModel::update_yield_stress(double k) {
    return yield_stress * ( (1+a)*exp(-b*k) - a*exp(-2*b*k) );
    // return yield_stress;
}

double MaterialModel::compute_derivative_yield_func_plastic_strain(double k) {
    return yield_stress * ( -b*(1+a)*exp(-b*k) + 2*b*a*exp(-2*b*k) );
    // return 0.0;
}

double MaterialModel::compute_von_mises_stress(SymmetricTensor<2, 2> stress_new) {
    SymmetricTensor<2, 2> stress_dev = deviator(stress_new);
    return sqrt(1.5 * stress_dev * stress_dev);
}

double MaterialModel::compute_yeild_func_val(double stress_von_mises, double stress_yield_new) {
    return stress_von_mises - stress_yield_new;
}

SymmetricTensor<2, 2>
MaterialModel::compute_direction_of_plastic_strain(SymmetricTensor<2, 2> stress_new, double stress_von_mises) {
    SymmetricTensor<2, 2> stress_dev = deviator(stress_new);
    return 1.5 * stress_dev / stress_von_mises;
}

SymmetricTensor<4, 2>
MaterialModel::compute_derivative_direction_of_plastic_strain_stress(SymmetricTensor<2, 2> m,
                                                                     double s_vm) {
    SymmetricTensor<4, 2> tmp = 1.5 * deviator_tensor<2>();
    tmp -= outer_product(m, m);
    tmp /= s_vm;
    return tmp;
}

double MaterialModel::compute_over_stress_coef(double f, double y_new) {
    return pow((f/y_new), n);
}

double
MaterialModel::compute_derivative_over_stress_coef_yield_func_val(double f, double y_new) {
    return (n/y_new) * pow((f/y_new), n-1);
}