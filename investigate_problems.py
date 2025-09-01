import re
import time
from pathlib import Path

from acados_template.acados_ocp_iterate import AcadosOcpFlattenedIterate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from acados_template.acados_ocp_batch_solver import AcadosOcpBatchSolver
from leap_c.examples import create_controller

def extract_status_and_index(filename):
    match = re.search(r"Status(\d+)", filename)
    if match:
        status = int(match.group(1))
    else:
        raise Exception("it should always match")

    index_match = re.search(r"problem_(\d+)", filename)
    index = int(index_match.group(1)) if index_match else float("inf")

    return status, index


def group_files_by_status(files_sorted):
    grouped_files = {1: [], 2: [], 4: []}

    for file in files_sorted:
        status, index = extract_status_and_index(file.name)
        if status in grouped_files:
            grouped_files[status].append(file)

    return grouped_files

def load_iterate_from_npz(npz_path: str | Path) -> AcadosOcpFlattenedIterate:
    data = np.load(npz_path, allow_pickle=True)
    
    return AcadosOcpFlattenedIterate(
        x=data['x'],
        u=data['u'],
        z=data['z'] if data["z"].size > 0 else np.empty((0,0)),
        sl=data['sl'] if data["sl"].size > 0 else np.empty((0,0)),
        su=data['su'] if data["su"].size > 0 else np.empty((0,0)),
        pi=data['pi'],
        lam=data['lam']
    )

def load_and_plot_all(files_grouped, identifier: str):
    for status in files_grouped.keys():
        data_list = []
        for i in range(len(files_grouped[status])):
            data_path = files_grouped[status][i]
            if identifier in data_path.name:
                data = np.load(data_path)
                data_list.append(data)
        if len(data_list) == 0:
            print("No examples found for status ", status)
            continue

        data_array = np.array(data_list)
        assert data_array.ndim == 2
        num_dimensions = data_array.shape[1]

        if num_dimensions == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))
            fig.suptitle(f"Histograms for Status {status}", fontsize=16)

            sns.histplot(data_array[:, 0], kde=False, ax=axes, color="blue")  # type:ignore
            axes.set_title("Histogram for Dimension 0")  # type:ignore
            axes.set_xlabel("Value")  # type:ignore
            axes.set_ylabel("Frequency")  # type:ignore

            plt.tight_layout()
        else:
            fig, axes = plt.subplots(num_dimensions, 1, figsize=(12, 6))
            fig.suptitle(f"Histograms for Status {status}", fontsize=16)

            for dim in range(num_dimensions):
                sns.histplot(data_array[:, dim], kde=False, ax=axes[dim], color="blue")  # type:ignore
                axes[dim].set_title(f"Histogram for Dimension {dim}")  # type:ignore
                axes[dim].set_xlabel("Value")  # type:ignore
                axes[dim].set_ylabel("Frequency")  # type:ignore

            plt.tight_layout()
        plots = Path.cwd() / "plots"
        plots.mkdir(exist_ok=True)
        plt.savefig(
            plots / f"Histograms_for_Status_{status}_identifier_'{identifier}'.png"
        )

def run_problem_instance(instance_status, instance_index_ls, files_grouped, single_solver):
    instance = files_grouped[instance_status][instance_index_ls]
    input = np.load(instance)
    x0 = input["x0"]
    param = input["params"]
    # print("X0: ", x0)
    # print("Param: ", param)
    iterate = load_iterate_from_npz(instance)
    
    single_solver.load_iterate_from_flat_obj(iterate)
    single_solver.set_p_global_and_precompute_dependencies(param.astype(np.float64))
    single_solver.solve_for_x0(
        x0.astype(np.float64),
        fail_on_nonzero_status=False,
        print_stats_on_failure=False,
    )
    # print(
    #     single_solver.get_status(),
    #     single_solver.get_residuals(),
    #     single_solver.get_stats("nlp_iter"),
    # )
    # single_solver.print_statistics()
    return single_solver.get_status(), single_solver.get_stats("nlp_iter")

if __name__ == "__main__":
    controller_name = "cartpole"
    controller = create_controller(controller_name, 
                                   #reuse_code_dir=Path.cwd()
                                   )
    load_path = Path.cwd() / "scripts" / "output" / "batch_solver_retry_CP_N=20_T=1_HESS=EX"
    ocp_solver: AcadosOcpBatchSolver = controller.diff_mpc.diff_mpc_fun.forward_batch_solver

    iterator = load_path.iterdir()
    files = list(iterator)

    print("Number of non convergences in directory: ", len(files))
    files_grouped = group_files_by_status(files)
    for group in files_grouped.keys():
        files_grouped[group] = sorted(
            files_grouped[group], key=lambda f: extract_status_and_index(f.name)
        )

    assert len(files) == len(files_grouped[1]) + len(files_grouped[2]) + len(
        files_grouped[4]
    ), "Some files have been lost?"
    print("No. of files grouped by status: ", [len(files_grouped.get(key, [])) for key in range(max(files_grouped.keys()) + 1)])

    instance_status = 4 # NOTE: CHANGE THIS FOR TRYING TO SOLVE INSTANCES WITH DIFFERENT STATUS
    instance_index_ls = int(len(files_grouped[instance_status]))
    statuses = 5 * [0]
    iters = []

    n_samples = min(10000, instance_index_ls)
    start = time.perf_counter()
    for i in range(n_samples):
        print("Progress: ", i, "/", n_samples)
        status, nlp_iter = run_problem_instance(
            instance_status=instance_status,
            instance_index_ls=i,
            files_grouped=files_grouped,
            single_solver=ocp_solver.ocp_solvers[0]
        )
        statuses[status] += 1
        iters.append(nlp_iter)
    end = time.perf_counter()

    print("Samples previous status: ", instance_status)
    print("Samples: ", instance_index_ls)
    print("Solves: ", n_samples)
    print("Time taken: ", end - start)
    print("Statuses: ", statuses)
    iters = np.array(iters)
    print("Iters_mean: ", iters.mean())
    print("Iters_var: ", iters.std())

    # load_and_plot_all(
    #     files_grouped=files_grouped,
    #     identifier="x0",
    # )
    # load_and_plot_all(
    #     files_grouped=files_grouped,
    #     identifier="param",
    # )

    # batch_solver_retry_CP_N=20_T=1_HESS=GN ====================================
    # Set qp_iter = 400 AND EXACT hessian (the latter by accident).
    # Samples previous status:  2
    # Samples:  2358
    # Solves:  2358
    # Time taken:  86.69235445605591
    # Statuses:  [1537, 0, 64, 0, 757]
    # Iters_mean:  8.751484308736217
    # Iters_var:  15.637355095640642
    # ===========================================================================
    # No additional settings
    # Samples previous status:  2
    # Samples:  2358
    # Solves:  2358
    # Time taken:  187.60468966141343
    # Statuses:  [0, 0, 2338, 0, 20]
    # Iters_mean:  99.1687871077184
    # Iters_var:  8.98709900753859
    # ===========================================================================
    # Setting max qp iter to 400
    # Samples previous status:  2
    # Samples:  2358
    # Solves:  2358
    # Time taken:  185.73385101975873
    # Statuses:  [2, 0, 2322, 0, 34]
    # Iters_mean:  98.81636980491942
    # Iters_var:  10.106504261233372
    # Weirdly, the time taken has not increased.
    # ===========================================================================
    # Setting max qp iter to 400 AND FUNNEL L1 globalization
    # Samples previous status:  2
    # Samples:  2358
    # Solves:  2358
    # Time taken:  180.18724164972082
    # Statuses:  [223, 0, 2115, 0, 20]
    # Iters_mean:  94.91051738761662
    # Iters_var:  16.58514565734388
    # ==========================================================================
    # Setting max qp iter to 400 AND FUNNEL L1 globalization and LM-reg = 1e-6
    # Samples previous status:  2
    # Samples:  2358
    # Solves:  2358
    # Time taken:  206.0463082930073
    # Statuses:  [262, 0, 2075, 0, 21]
    # Iters_mean:  94.30873621713316
    # Iters_var:  17.167923322095582
    # ==========================================================================
    # The above AND
    # opts.qp_solver_tol_comp = 1e-8
    # opts.qp_solver_tol_stat = 1e-8
    # opts.qp_solver_tol_eq = 1e-8
    # opts.qp_solver_tol_ineq = 1e-8
    # opts.qp_solver_mu0 = 1e6
    # opts.qp_solver_t0_init = 0
    # Samples previous status:  2
    # Samples:  2358
    # Solves:  2358
    # Time taken:  214.13987614028156
    # Statuses:  [262, 0, 2076, 0, 20]
    # Iters_mean:  94.77947413061916
    # Iters_var:  16.499483716547154
    # ==========================================================================
    # EVERYTHING except L1 funnel and no levenberg, but with anderson acceleration and 300 nlp iters.
    # Samples previous status:  2
    # Samples:  2358
    # Solves:  2358
    # Time taken:  244.42134702019393
    # Statuses:  [1809, 0, 479, 0, 70]
    # Iters_mean:  115.80449533502969
    # Iters_var:  108.18041835046849
    #===========================================================================
    # As above but with 100 nlp iters
    # Samples previous status:  2
    # Samples:  2358
    # Solves:  2358
    # Time taken:  140.8446823116392
    # Statuses:  [1550, 0, 752, 0, 56]
    # Iters_mean:  63.65224766751484
    # Iters_var:  29.98720990493582
    
    
    
    
    # AS EXPECTED, THERE WERE ONLY 22 INSTANCES OF STATUS 4 here, and on a rerun, they all hit status 2
    # batch_solver_retry_CP_N=20_T=1_HESS=EX ===================================
    # AS EXPECTED, THERE WERE ONLY 18 INSTANCES OF STATUS 2 here, and on a rerun, 1 succeeded, 3 went status 2, the rest went status 4
    #===========================================================================
    # Setting max qp iter to 400
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  85.95564376376569
    # Statuses:  [27, 0, 15, 0, 5007]
    # Iters_mean:  1.5137651020003962
    # Iters_var:  5.508906225075426
    #===========================================================================
    # Setting max qp iter to 400 and FUNNEL L1 PEN
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  44.704310880973935
    # Statuses:  [72, 0, 0, 8, 4969]
    # Iters_mean:  1.4246385422856012
    # Iters_var:  2.836034676226176
    #===========================================================================
    # Setting max qp iter to 400 and FUNNEL L1 PEN and solver is SQP WITH FEASIBLE QP
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  44.57927850447595
    # Statuses:  [41, 0, 0, 0, 5008]
    # Iters_mean:  0.3111507229154288 => it fails faster? :D
    # Iters_var:  2.2785862461414848
    #===========================================================================
    # Setting max qp iter to 400 and FUNNEL L1 PEN and solver is SQP WITH FEASIBLE QP
    # AND the options 
    # opts.qp_solver_tol_comp = 1e-8
    # opts.qp_solver_tol_stat = 1e-8
    # opts.qp_solver_tol_eq = 1e-8
    # opts.qp_solver_tol_ineq = 1e-8
    # opts.qp_solver_mu0 = 1e6
    # opts.qp_solver_t0_init = 0
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  255.59366230852902
    # Statuses:  [438, 0, 0, 0, 4611]
    # Iters_mean:  1.772826302238067
    # Iters_var:  3.9375807094795516
    #===========================================================================
    # Using qp iter 400, qp options as above, mirror regularization, DAQP as solver
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  62.20100602693856
    # Statuses:  [3506, 0, 1528, 0, 15]
    # Iters_mean:  62.831055654585064
    # Iters_var:  26.887428783616524
    #===========================================================================
    # As above with nlp iters 200
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  72.58136547915637
    # Statuses:  [4119, 0, 915, 0, 15]
    # Iters_mean:  85.01604278074866
    # Iters_var:  62.42403456372445
    #===========================================================================
    # As above with nlp iters 500
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  110.24579677172005
    # Statuses:  [4414, 0, 620, 0, 15]
    # Iters_mean:  129.19726678550208
    # Iters_var:  156.42719306596746
    #==========================================================================
    # As above but without mirror regularization
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  9.243193548172712
    # Statuses:  [1, 0, 0, 0, 5048]
    # Iters_mean:  1.0570409982174689
    # Iters_var:  0.27346220767813145
    #==========================================================================
    # As above but with PROJECT regularization
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  136.21702730469406
    # Statuses:  [4649, 0, 384, 2, 14]
    # Iters_mean:  93.35333729451376
    # Iters_var:  141.62875494694103
    #==========================================================================
    # As above but with MERIT_BACKTRACKING globalization
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  89.89936039689928
    # Statuses:  [4564, 0, 460, 0, 25]
    # Iters_mean:  96.40879382055853
    # Iters_var:  146.46601778273546
    #==========================================================================
    # As above but without globalization
    # Samples previous status:  4
    # Samples:  5049
    # Solves:  5049
    # Time taken:  48.08923596236855
    # Statuses:  [3905, 1, 198, 0, 945]
    # Iters_mean:  60.60507031095266
    # Iters_var:  111.99162153376095