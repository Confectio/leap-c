import time

from acados_template.acados_ocp_batch_solver import AcadosOcpBatchSolver
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedIterate, AcadosOcpFlattenedBatchIterate
import numpy as np

from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer
from leap_c.ocp.acados.utils.prepare_solver import prepare_batch_solver
from leap_c.ocp.acados.data import AcadosOcpSolverInput
from pathlib import Path

cwd = Path.cwd()
NONSOLVED_PROBLEMS_PATH = cwd / "output" / "batch_solver_retry_CP_N=20_T=1_HESS=EX"
NONSOLVED_PROBLEMS_PATH.mkdir(parents=True, exist_ok=True)
NONSOLVED_PROBLEM_COUNTER = 0
def save_problems_if_status_nonzero(solver_input: AcadosOcpSolverInput, inp_iterates: AcadosOcpFlattenedBatchIterate, status: np.ndarray) -> None:
    global NONSOLVED_PROBLEM_COUNTER
    global NONSOLVED_PROBLEMS_PATH
    for idx, stat in enumerate(status):
        if stat != 0:
            problem_data = {
                "x0": solver_input.x0[idx],
                #NOTE: Assumes this is the only input that matters, 
                # in particular also no p stagewise.
                "params": solver_input.get_sample(idx).p_global,
                "status": stat,
                "x": inp_iterates.x[idx],
                "u": inp_iterates.u[idx],
                "z": inp_iterates.z[idx] if inp_iterates.z.size > 0 else np.array([]),
                "sl": inp_iterates.sl[idx] if inp_iterates.sl.size > 0 else np.array([]),
                "su": inp_iterates.su[idx] if inp_iterates.su.size > 0 else np.array([]),
                "pi": inp_iterates.pi[idx],
                "lam": inp_iterates.lam[idx],
            }
            problem_path = NONSOLVED_PROBLEMS_PATH / f"problem{NONSOLVED_PROBLEM_COUNTER}_Status{stat}.npz"
            np.savez(problem_path, **problem_data)
            NONSOLVED_PROBLEM_COUNTER += 1


def solve_with_retry(
    batch_solver: AcadosOcpBatchSolver,
    initializer: AcadosDiffMpcInitializer,
    ocp_iterate: AcadosOcpFlattenedBatchIterate | None,
    solver_input: AcadosOcpSolverInput,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve a batch of ocps, and retries in case of divergence.

    This function prepares the batch solver by loading the iterate, setting
    the initial conditions, and configuring the global and stage-wise
    parameters. If `p_global` or `p_stagewise` is not provided, it will
    check if the model has default parameters and load them accordingly.

    Args:
        batch_solver: The batch solver to use.
        initializer: The initializer used for retries.
        ocp_iterate: The iterate to load into the batch solver.
        solver_input: The input data for the solver, which includes initial
            conditions and parameters.

    Returns:
        The solving stats.
    """
    batch_size = solver_input.batch_size

    if ocp_iterate is None:
        ocp_iterate = initializer.batch_iterate(solver_input)
        with_retry = False
    else:
        with_retry = True

    prepare_batch_solver(batch_solver, ocp_iterate, solver_input)
    inp_iterates = batch_solver.store_iterate_to_flat_obj(n_batch=batch_size)
    
    start = time.perf_counter()
    batch_solver.solve(n_batch=batch_size)
    time_solve = time.perf_counter() - start

    active_solvers = batch_solver.ocp_solvers[:batch_size]
    batch_status = np.array([solver.status for solver in active_solvers])

    if with_retry and any(status != 0 for status in batch_status):

        for idx, solver in enumerate(active_solvers):
            if batch_status[idx] == 0:
                continue
            single_iterate = initializer.single_iterate(solver_input.get_sample(idx))
            solver.load_iterate_from_flat_obj(single_iterate)

        start_retry = time.perf_counter()
        batch_solver.solve(n_batch=batch_size)
        time_solve += time.perf_counter() - start_retry

    batch_status_retry = np.array([solver.status for solver in active_solvers])
    # Probably the problems that cant be solved by the retry are the most interesting ones
    save_problems_if_status_nonzero(solver_input, inp_iterates, batch_status_retry)
    stats = {
        "solving_time": time_solve,
        "success_rate": (batch_status_retry == 0).mean(),
        "retry_rate": (batch_status != 0).mean(),
    }

    return batch_status_retry, stats
