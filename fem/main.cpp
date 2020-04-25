#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/timer.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <fstream>
#include <iostream>
#include <sys/stat.h>

using namespace dealii;


double delta_func(unsigned int i, unsigned int j)
{
    return (i==j) ? 1.0 : 0.0;
}

//======================================================
//      Point history
//======================================================
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

double PointHistory::get_strain_plastic_cum() const {
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


//======================================================
//      Material model
//======================================================
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

//    std::cout << "yield_func_val " <<  yield_func_val << std::endl;
//    std::cout << "strain_plastic_cum_old " <<  strain_plastic_cum_old << std::endl;
//    std::cout << "strain_delta " <<  strain_delta << std::endl;
//    std::cout << std::endl;

    // yield_func_val = -1;
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

//    std::cout << "stress_old " <<  stress_old << std::endl;
//    std::cout << "stress_new " <<  stress_new << std::endl;
//    std::cout << "strain_delta " <<  strain_delta << std::endl;
//    std::cout << std::endl;

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




//======================================================
//      Model
//======================================================
class Model
{
public:
    Model ( std::map<int, std::map<std::string, double>> material_config_,
            std::map<std::string, std::string> path_config_);
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
};

Model::Model(std::map<int, std::map<std::string, double>> material_config_,
             std::map<std::string, std::string> path_config_):
    fe(FE_Q<2>(1), 2),
    dof_handler(triangulation),
    quadrature_formula(fe.degree + 1),
    timestep_no(0),
    n_timestep(400)
{
    path_config = path_config_;

    material_models[1] = new MaterialModel(material_config_[1]);
    material_models[2] = new MaterialModel(material_config_[2]); // weak material
}

void Model::update_quadrature_point_history()
{
    const unsigned int n_q_points = quadrature_formula.size();

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const std::vector<std::shared_ptr<PointHistory>> history = quadrature_point_history.get_data(cell);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            history[q_point]->update_history();
    }
}

void Model::solve_linear_problem()
{
    SolverControl solver_control(dof_handler.n_dofs(), 1e-8);
    SolverCG<> solver(solver_control);

    solver.solve(system_matrix, tmp_displacement, system_rhs, PreconditionIdentity());
    incremental_displacement += tmp_displacement;
}

void Model::assemble_system()
{
    system_rhs    = 0;
    system_matrix = 0;

    FEValues<2> fe_values(
        fe,
        quadrature_formula,
        update_values | update_gradients | update_JxW_values);

    const unsigned int                      dofs_per_cell = fe.dofs_per_cell;
    const unsigned int                      n_q_points    = quadrature_formula.size();
    FullMatrix<double>                      cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>                          cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector displacement(0);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        MaterialModel *material_model = material_models.find(cell->material_id())->second;

        std::vector<SymmetricTensor<2, 2>> strain_increment_tensor(n_q_points);
        fe_values[displacement].get_function_symmetric_gradients(incremental_displacement, strain_increment_tensor);

        const std::vector<std::shared_ptr<PointHistory>> history_vec = quadrature_point_history.get_data(cell);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
//            std::cout << q_point << std::endl;
            std::shared_ptr<PointHistory> history = history_vec[q_point];
            SymmetricTensor<2, 2> strain_delta = strain_increment_tensor[q_point];

            SymmetricTensor<2, 2> stress;
            SymmetricTensor<4, 2> stiffness;
            material_model->update(strain_delta, history, stress, stiffness);

//            std::cout << strain_delta << std::endl;
//            std::cout << history->get_strain_plastic_cum() << std::endl;

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                SymmetricTensor<2, 2> b_tensor_i = fe_values[displacement].symmetric_gradient(i, q_point);
                cell_rhs(i) += - b_tensor_i * stress * fe_values.JxW(q_point);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    SymmetricTensor<2, 2> b_tensor_j = fe_values[displacement].symmetric_gradient(j, q_point);
                    cell_matrix(i, j) += b_tensor_i * stiffness * b_tensor_j  * fe_values.JxW(q_point);
                }
            }
        }

//        std::cout << std::endl;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                system_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i, j));
        }

        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
    }
}

void Model::apply_boundary(bool is_first_iteration)
{
    FEValuesExtractors::Scalar x_component(0);
    FEValuesExtractors::Scalar y_component(1);
    std::map<types::global_dof_index, double> boundary_values;

    VectorTools::interpolate_boundary_values(
            dof_handler,
            3,
            Functions::ZeroFunction<2>(2),
            boundary_values,
            fe.component_mask(x_component));

    VectorTools::interpolate_boundary_values(
            dof_handler,
            4,
            Functions::ZeroFunction<2>(2),
            boundary_values,
            fe.component_mask(y_component));

    if (is_first_iteration)
        VectorTools::interpolate_boundary_values(
                dof_handler,
                6,
                Functions::ConstantFunction<2>(-0.005, 2),
                boundary_values,
                fe.component_mask(y_component));
    else
        VectorTools::interpolate_boundary_values(
                dof_handler,
                6,
                Functions::ZeroFunction<2>(2),
                boundary_values,
                fe.component_mask(y_component));

    MatrixTools::apply_boundary_values(
            boundary_values,
            system_matrix,
            tmp_displacement,
            system_rhs);

//    std::cout << "boundary ids" << std::endl;
//    for (auto &cell : dof_handler.active_cell_iterators())
//    {
//        std::cout << "cell id" << cell->id() << std::endl;
//        for (int i=0; i<4; i++)
//        {
//            std::cout << cell->face(i)->boundary_id() << std::endl;
//        }
//    }

//    FEValuesExtractors::Scalar x_component(0);
//    std::map<types::global_dof_index, double> boundary_values;
//
//    VectorTools::interpolate_boundary_values(
//            dof_handler,
//            6,
//            Functions::ZeroFunction<2>(2),
//            boundary_values,
//            fe.component_mask(x_component));
//
//    if (is_first_iteration)
//        VectorTools::interpolate_boundary_values(
//                dof_handler,
//                4,
//                Functions::ConstantFunction<2>(0.01, 2),
//                boundary_values,
//                fe.component_mask(x_component));
//    else
//        VectorTools::interpolate_boundary_values(
//                dof_handler,
//                4,
//                Functions::ZeroFunction<2>(2),
//                boundary_values,
//                fe.component_mask(x_component));

    MatrixTools::apply_boundary_values(
            boundary_values,
            system_matrix,
            tmp_displacement,
            system_rhs);


//    std::cout << std::endl;
//    for (const auto &[k, v] : boundary_values)
//        std::cout << k << " " << v << std::endl;
}

bool Model::solve_newton()
{
    double residual(1e10);
    bool is_first_iteration(true);
    int n_iterations(0);
    Timer timer_assemble_system;
    Timer timer_solver;
    Timer timer_time_step;

    timer_time_step.start();
    while (residual > 1e-8)
    {
        if (n_iterations>20)
            throw;
        n_iterations++;

        timer_assemble_system.start();
        assemble_system();
        timer_assemble_system.stop();

        apply_boundary(is_first_iteration);

        timer_solver.start();
        solve_linear_problem();
        timer_solver.stop();

        residual = system_rhs.l2_norm();
        std::cout << "\t\titer: " << n_iterations << ", residual: " << residual << std::endl;
        is_first_iteration = false;
    }
    timer_time_step.stop();

    std::cout << "time step      time: " << timer_time_step.wall_time() << std::endl;
    std::cout << "assemle system time: " << timer_assemble_system.wall_time() << std::endl;
    std::cout << "linear solver  time: " << timer_solver.wall_time() << std::endl;
    std::cout << std::endl;

    return true;
}

void Model::do_timestep()
{
    while (timestep_no < n_timestep)
    {
        ++timestep_no;
        std::cout << "Timestep: " << timestep_no << std::endl;

        bool converged = solve_newton();

        if (converged)
        {
            std::cout << "Updating quadrature point data..." << std::flush;
            update_quadrature_point_history();
            output_results();

            total_displacement += incremental_displacement;
            incremental_displacement = 0.0;
        }

        std::cout << std::endl << std::endl;
    }
}

void Model::output_results() {
    // write solution to vtu
    DataOut<2> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
            2,
            DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_names(2, "displacement");

    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(
            total_displacement,
            solution_names,
            DataOut<2>::type_dof_data,
            data_component_interpretation);
    data_out.build_patches();

    path_config["i"];
    path_config["output_solution"];

    std::string output_path_solution = path_config["output_solution"]
                                        + "solution_"
                                        + std::to_string(timestep_no)
                                        + ".vtu";

    std::ofstream output(output_path_solution);
    data_out.write_vtu(output);


    //------------------------------------------------------
    //          write plastic strain to
    //------------------------------------------------------
    FE_Q<2>       history_fe(1);
    DoFHandler<2> history_dof_handler(triangulation);
    history_dof_handler.distribute_dofs(history_fe);

    Vector<double>
            plastic_strain_field(history_dof_handler.n_dofs()),
            local_plastic_strain_values_at_qpoints(quadrature_formula.size()),
            local_plastic_strain_fe_values(history_fe.dofs_per_cell);

    FullMatrix<double> qpoint_to_dof_matrix(history_fe.dofs_per_cell, quadrature_formula.size());

    FETools::compute_projection_from_quadrature_points_matrix(
            history_fe,
            quadrature_formula, quadrature_formula,
            qpoint_to_dof_matrix);

    for (const auto &cell : history_dof_handler.active_cell_iterators())
    {
        const std::vector<std::shared_ptr<PointHistory>> history_vec = quadrature_point_history.get_data(cell);

        for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
            local_plastic_strain_values_at_qpoints(q) = history_vec[q]->get_strain_plastic_cum();

        qpoint_to_dof_matrix.vmult(local_plastic_strain_fe_values, local_plastic_strain_values_at_qpoints);

        cell->set_dof_values(local_plastic_strain_fe_values, plastic_strain_field);
    }

    FE_Q<2>       fe_1 (1);
    DoFHandler<2> dof_handler_1 (triangulation);
    dof_handler_1.distribute_dofs (fe_1);

    Vector<double>
            plastic_strain_on_vertices (dof_handler_1.n_dofs()),
            counter_on_vertices (dof_handler_1.n_dofs());

    plastic_strain_on_vertices = 0;
    counter_on_vertices = 0;

    for (const auto &cell : dof_handler_1.active_cell_iterators())
    {
        cell->get_dof_values (plastic_strain_field, local_plastic_strain_fe_values);

        for  (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
        {
            types::global_dof_index dof_1_vertex = cell->vertex_dof_index(v, 0);

            counter_on_vertices (dof_1_vertex) += 1;
            plastic_strain_on_vertices (dof_1_vertex) += local_plastic_strain_fe_values (v);
        }
    }

    for (unsigned int id=0; id<dof_handler_1.n_dofs(); ++id)
        plastic_strain_on_vertices(id) /= counter_on_vertices(id);

    DataOut<2> data_out_p;
    data_out_p.attach_dof_handler(dof_handler_1);
    data_out_p.add_data_vector(plastic_strain_on_vertices, "plastic_strain");
    data_out_p.build_patches();

    std::string path = path_config["output_plastic_strain"]
                            + "plastic_strain_"
                            + std::to_string(timestep_no)
                            + ".vtu";
    std::ofstream output_p(path);
    data_out_p.write_vtu(output_p);
}

void Model::setup_system()
{
    dof_handler.distribute_dofs(fe);

    std::cout
            << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    incremental_displacement.reinit(dof_handler.n_dofs());
    total_displacement.reinit(dof_handler.n_dofs());
    tmp_displacement.reinit(dof_handler.n_dofs());

    total_displacement = 0.0;

    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        quadrature_formula.size());
}

void Model::make_grid()
{
    GridIn<2> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f(path_config["msh"]);
    gridin.read_msh(f);

    std::cout
            << "Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;

    std::cout
            << "Number of active lines: "
            << triangulation.n_active_lines()
            << std::endl;

    std::cout
            << "Number of vertices: "
            << triangulation.n_vertices()
            << std::endl;
}

void Model::run()
{
    make_grid();
    setup_system();
    do_timestep();
}


void tokenize(std::string const &str, const char delim, std::vector<std::string> &out)
{
    // construct a stream from the string
    std::stringstream ss(str);

    std::string s;
    while (std::getline(ss, s, delim)) {
        out.push_back(s);
    }
}

void parse_arg(
        int argc, char *argv[],
        std::map<int, std::map<std::string, double>> &config)
{
    std::vector<int> material_ids = {1, 2};
    std::vector<std::string> material_param_names = {"e", "b", "y"};

    for (int material_id : material_ids)
    {
        std::map<std::string, double> config_mat;
        for (const std::string& material_param_name : material_param_names)
        {
            for (int i=1; i<argc; i++)
            {
                if (std::string(argv[i]).find("_" + std::to_string(material_id)) != std::string::npos)
                    if (std::string(argv[i]).find(material_param_name) != std::string::npos)
                    {
                        std::vector<std::string> out;
                        tokenize(argv[i], '=', out);
                        config_mat.insert(std::make_pair(material_param_name, std::stod(out[1])));
                    }
            }
        }

        config.insert(std::make_pair(material_id, config_mat));
    }

    std::cout << "Material config:" << std::endl;
    for (const auto &pair : config)
    {
        for (const auto &pair_mat : pair.second)
        {
            std::cout
                << "material_id: " << pair.first
                << ", " << pair_mat.first
                << " : " << pair_mat.second << std::endl;
        }
    }
}


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

    material_config_1["b"] = std::stod(argv[4]);
    material_config_1["y"] = std::stod(argv[5]);

    material_config_2["b"] = std::stod(argv[6]);
    material_config_2["y"] = std::stod(argv[7]);

    material_config[1] = material_config_1;
    material_config[2] = material_config_2;

    Model model(material_config, path_config);
    model.run();
}




//======================================================
//      Incremental boundary value
//======================================================
//class IncrementalBoundaryValues: public Function<2>
//{
//public:
//    IncrementalBoundaryValues (double present_time,
//                               double end_time);
//    virtual
//    void
//    vector_value (const Point<2> &p,
//                  Vector<double>   &values) const;
//    virtual
//    void
//    vector_value_list (const std::vector<Point<2> > &points,
//                       std::vector<Vector<double> >   &value_list) const;
//
//private:
//    const double present_time;
//    const double end_time;
//    const double imposed_displacement;
//};
//
//IncrementalBoundaryValues::IncrementalBoundaryValues (
//    const double present_time,
//    const double end_time):
//        Function<2> (2),
//        present_time (present_time),
//        end_time (end_time),
//        imposed_displacement (0.001)
//{}
//
//void IncrementalBoundaryValues::vector_value (
//    const Point<2> &p,
//    Vector<double> &values) const
//{
//    AssertThrow (values.size() == 2,
//                 ExcDimensionMismatch (values.size(), 2));
//
//    values = 0.;
//    values(0) = imposed_displacement;
//}
//
//void IncrementalBoundaryValues::vector_value_list(const std::vector<Point<2> > &points,
//                                                  std::vector<Vector<double> > &value_list) const {
//    Function::vector_value_list(points, value_list);
//}