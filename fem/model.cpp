#include <map>
#include <string>
#include <fstream>
#include <iostream>

#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/timer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include "model.h"
#include "material_model.h"



Model::Model(std::map<int, std::map<std::string, double>> material_config_,
             std::map<std::string, std::string> path_config_,
             int n_timestep_,
             int verbose_):
    fe(FE_Q<2>(1), 2),
    dof_handler(triangulation),
    quadrature_formula(fe.degree + 1),
    timestep_no(0)
{
    n_timestep = n_timestep_;
    path_config = path_config_;
    verbose = verbose_;

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
            std::shared_ptr<PointHistory> history = history_vec[q_point];
            SymmetricTensor<2, 2> strain_delta = strain_increment_tensor[q_point];

            SymmetricTensor<2, 2> stress;
            SymmetricTensor<4, 2> stiffness;
            material_model->update(strain_delta, history, stress, stiffness);

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
                5,
                Functions::ConstantFunction<2>(0.01, 2),
                boundary_values,
                fe.component_mask(x_component));
    else
        VectorTools::interpolate_boundary_values(
                dof_handler,
                5,
                Functions::ZeroFunction<2>(2),
                boundary_values,
                fe.component_mask(x_component));

    MatrixTools::apply_boundary_values(
            boundary_values,
            system_matrix,
            tmp_displacement,
            system_rhs);

    MatrixTools::apply_boundary_values(
            boundary_values,
            system_matrix,
            tmp_displacement,
            system_rhs);

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
        if (n_iterations>20) {
            if (verbose==1)
                std::cout << "[WARNING] didn't converge!" << std::endl;
            break;
        }
        n_iterations++;

        timer_assemble_system.start();
        assemble_system();
        timer_assemble_system.stop();

        apply_boundary(is_first_iteration);

        timer_solver.start();
        solve_linear_problem();
        timer_solver.stop();

        residual = system_rhs.l2_norm();
        if (verbose==1)
            std::cout << "\t\titer: " << n_iterations << ", residual: " << residual << std::endl;
        is_first_iteration = false;
    }
    timer_time_step.stop();

    if (verbose==1)
    {
        std::cout << "time step      time: " << timer_time_step.wall_time() << std::endl;
        std::cout << "assemle system time: " << timer_assemble_system.wall_time() << std::endl;
        std::cout << "linear solver  time: " << timer_solver.wall_time() << std::endl;
        std::cout << std::endl;
    }
    

    return true;
}

void Model::do_timestep()
{
    while (timestep_no < n_timestep)
    {
        ++timestep_no;
        if (verbose==1)
            std::cout 
                << "Timestep: " 
                << timestep_no
                << " out of " 
                << n_timestep
                << std::endl;

        bool converged = solve_newton();

        if (converged)
        {
            if (verbose==1)
                std::cout << "Updating quadrature point data..." << std::flush;
            update_quadrature_point_history();
            output_results();

            total_displacement += incremental_displacement;
            incremental_displacement = 0.0;
        }

        if (verbose==1)
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

    Vector<double>
            vm_stress_field(history_dof_handler.n_dofs()),
            local_vm_stress_values_at_qpoints(quadrature_formula.size()),
            local_vm_stress_fe_values(history_fe.dofs_per_cell);

    FullMatrix<double> qpoint_to_dof_matrix(history_fe.dofs_per_cell, quadrature_formula.size());

    FETools::compute_projection_from_quadrature_points_matrix(
            history_fe,
            quadrature_formula, quadrature_formula,
            qpoint_to_dof_matrix);

    for (const auto &cell : history_dof_handler.active_cell_iterators())
    {
        const std::vector<std::shared_ptr<PointHistory>> history_vec = quadrature_point_history.get_data(cell);

        for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        {
            SymmetricTensor<2, 2> stress = history_vec[q]->get_stress_old();
            double von_Mises_stress = std::sqrt(
                stress[0][0] * stress[0][0] + 
                stress[1][1] * stress[1][1] - 
                stress[0][0] * stress[1][1] + 
                3 * stress[0][1] * stress[0][1]);

            local_vm_stress_values_at_qpoints(q) = von_Mises_stress;

            local_plastic_strain_values_at_qpoints(q) = history_vec[q]->get_strain_plastic_cum();
        }
        qpoint_to_dof_matrix.vmult(local_plastic_strain_fe_values, local_plastic_strain_values_at_qpoints);
        qpoint_to_dof_matrix.vmult(local_vm_stress_fe_values, local_vm_stress_values_at_qpoints);

        cell->set_dof_values(local_plastic_strain_fe_values, plastic_strain_field);
        cell->set_dof_values(local_vm_stress_fe_values, vm_stress_field);
    }

    FE_Q<2>       fe_1 (1);
    DoFHandler<2> dof_handler_1 (triangulation);
    dof_handler_1.distribute_dofs (fe_1);

    Vector<double>
            plastic_strain_on_vertices (dof_handler_1.n_dofs()),
            counter_on_vertices (dof_handler_1.n_dofs());

    Vector<double>
            vm_stress_on_vertices (dof_handler_1.n_dofs());

    plastic_strain_on_vertices = 0;
    counter_on_vertices = 0;
    vm_stress_on_vertices = 0;

    for (const auto &cell : dof_handler_1.active_cell_iterators())
    {
        cell->get_dof_values (plastic_strain_field, local_plastic_strain_fe_values);
        cell->get_dof_values (vm_stress_field, local_vm_stress_fe_values);

        for  (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
        {
            types::global_dof_index dof_1_vertex = cell->vertex_dof_index(v, 0);

            counter_on_vertices (dof_1_vertex) += 1;
            plastic_strain_on_vertices (dof_1_vertex) += local_plastic_strain_fe_values (v);
            vm_stress_on_vertices (dof_1_vertex) += local_vm_stress_fe_values (v);
        }
    }

    for (unsigned int id=0; id<dof_handler_1.n_dofs(); ++id)
    {
        plastic_strain_on_vertices(id) /= counter_on_vertices(id);
        vm_stress_on_vertices(id) /= counter_on_vertices(id);
    }

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


    DataOut<2> data_out_s;
    data_out_s.attach_dof_handler(dof_handler_1);
    data_out_s.add_data_vector(vm_stress_on_vertices, "vm_stress");
    data_out_s.build_patches();

    std::string path_vm = path_config["output_vm_stress"]
                            + "vm_stress_"
                            + std::to_string(timestep_no)
                            + ".vtu";
    std::ofstream out_s(path_vm);
    data_out_s.write_vtu(out_s);
}


void Model::setup_system()
{
    dof_handler.distribute_dofs(fe);

    if (verbose==1)
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

    if (verbose==1)
    {
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
}

void Model::run()
{
    make_grid();
    setup_system();
    do_timestep();
}