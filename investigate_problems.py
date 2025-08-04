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
    #     ocp_solver.get_status(),
    #     ocp_solver.get_residuals(),
    #     ocp_solver.get_stats("nlp_iter"),
    # )
    # ocp_solver.print_statistics()
    return single_solver.get_status(), single_solver.get_stats("nlp_iter")

if __name__ == "__main__":
    controller_name = "cartpole"
    controller = create_controller(controller_name, 
                                   #reuse_code_dir=Path.cwd()
                                   )
    load_path = Path.cwd() / "scripts" / "output" / "batch_solver_retry_CP_N=20_T=1_HESS=GN"
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

    instance_status = 4
    instance_index_ls = int(len(files_grouped[instance_status]))
    statuses = 5 * [0]
    iters = []

    n_samples = min(1000, instance_index_ls)
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

    # Possible options:
    # ocp.solver_options.nlp_solver_type = "SQP_WITH_FEASIBLE_QP"
    # ocp.solver_options.globalization = "FUNNEL_L1PEN_LINESEARCH"
    # ocp.solver_options.qp_solver_mu0 = 1e3
    # ocp.solver_options.qp_solver_iter_max = 1000
    # ocp.solver_options.nlp_solver_max_iter = 1000