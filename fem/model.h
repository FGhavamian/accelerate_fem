#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/quadrature_lib.h>
#ifndef MODEL_H
#define MODEL_H

#include <map>
#include <string>

#include "point_history.h"
#include "material_model.h"


using namespace dealii;


class Model
{
public:
    Model ( std::map<int, std::map<std::string, double>> material_config_,
            std::map<std::string, std::string> path_config_,
            int n_timestep_,
            int verbose_);
    void run ();

private:
    void make_grid ();
    void setup_system();
    void assemble_system ();
    void solve_linear_problem();
    void output_results ();
    void do_timestep();
    bool solve_newton();
    void update_quadrature_point_history();
    void apply_boundary(bool is_first_iteration);

    std::map<std::string, std::string> path_config;

    Triangulation<2>     triangulation;
    FESystem<2>          fe;
    DoFHandler<2>        dof_handler;
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double>       system_rhs;
    Vector<double>       incremental_displacement;
    Vector<double>       total_displacement;
    Vector<double>       tmp_displacement;
    const QGauss<2>      quadrature_formula;
    CellDataStorage<typename Triangulation<2>::cell_iterator, PointHistory> quadrature_point_history;

    std::map<int, MaterialModel *>  material_models;
    unsigned int         timestep_no;
    unsigned int         n_timestep;
    int                  verbose;
};

#endif