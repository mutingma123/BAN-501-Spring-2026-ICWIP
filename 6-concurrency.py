import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import concurrent.futures as concurrent_futures
    import multiprocessing
    import os
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns

    from _concurrency_helpers import CPU_BOUND_NUMBERS, is_prime, simulate_io_task

    sns.set_style("whitegrid")
    return (
        CPU_BOUND_NUMBERS,
        concurrent_futures,
        is_prime,
        mo,
        multiprocessing,
        np,
        os,
        pl,
        plt,
        simulate_io_task,
        sns,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Concurrency in Python: Threads and Processes

    Many tasks in data science involve waiting — waiting for API responses, waiting for database
    queries, or waiting for computations to finish. **Concurrency** lets us overlap these waits so
    programs finish faster.

    This notebook demonstrates:

    1. The difference between **threads** and **processes** in Python
    2. When each approach helps (and when it doesn't)
    3. How to use `concurrent.futures` for both

    By the end, you'll understand why scikit-learn's `n_jobs` parameter speeds up model training
    and when to reach for threads vs. processes in your own code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Terminology: Cores, Hardware Threads, and Software Threads

    The word **"thread"** means different things in different contexts, and this causes confusion.

    ### CPU Core

    A **core** is a physical processing unit on a chip. A dual-core processor has 2 independent
    units that can each execute instructions simultaneously.

    ### Hardware Threads (SMT / Hyper-Threading)

    Modern CPUs use **Simultaneous Multithreading (SMT)** — Intel brands this as
    **Hyper-Threading**. Each physical core presents itself as 2 **logical processors** to the
    operating system. When one hardware thread stalls (e.g., waiting for memory), the other can
    use the core's execution units.

    > A spec sheet that says **"2 cores, 4 threads"** means 2 physical cores × 2 hardware threads
    > each = 4 logical processors visible to the OS.

    ### Software Threads

    A **software thread** is an independent sequence of instructions scheduled by the operating
    system. A single program can create thousands of software threads regardless of how many
    hardware threads exist. The OS time-slices them across available logical processors.

    ### The Confusion

    Hardware marketing and software developers use **"thread"** to mean fundamentally different
    things:

    | Context | "Thread" means | Example |
    |---------|---------------|---------|
    | Hardware specs | Logical processor (SMT) | "4 cores, 8 threads" |
    | Software / this notebook | OS-scheduled execution stream | `ThreadPoolExecutor(max_workers=8)` |

    **This notebook focuses entirely on software threads and processes.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Concurrency vs. Parallelism and the GIL

    ### Concurrency vs. Parallelism

    - **Concurrency** = managing multiple tasks with overlapping lifetimes. The tasks may not run
      at the exact same instant — they can *interleave* on a single core.
    - **Parallelism** = executing multiple tasks *simultaneously* on different cores. Parallelism
      is a subset of concurrency.

    ### Python's Global Interpreter Lock (GIL)

    CPython (the standard Python interpreter) has a **Global Interpreter Lock (GIL)**: only one
    thread can execute Python bytecode at a time. However, the GIL is **released** during IO
    operations (network calls, file reads, `time.sleep`).

    This has practical consequences:

    | Approach | Mechanism | GIL impact | Best for |
    |----------|-----------|-----------|----------|
    | **Threads** (`ThreadPoolExecutor`) | Shared memory, lightweight | Only one thread runs Python at a time | IO-bound tasks (API calls, file IO) |
    | **Processes** (`ProcessPoolExecutor`) | Separate interpreters, true parallelism | Each process has its own GIL | CPU-bound tasks (computation) |

    We'll demonstrate both scenarios below.
    """)
    return


@app.cell
def _(multiprocessing, os):
    num_cores = os.cpu_count()
    print(f"os.cpu_count():              {num_cores}")
    print(f"multiprocessing.cpu_count(): {multiprocessing.cpu_count()}")
    print()
    print(
        "These report logical processors (physical cores × hardware threads per core)."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IO-Bound Tasks

    A task is **IO-bound** when it spends most of its time waiting for external resources:
    network responses, disk reads, database queries, etc.

    Real-world examples:
    - Calling a REST API for each row in a dataset
    - Reading hundreds of CSV files from disk
    - Querying a database for multiple tables

    We simulate IO-bound work with `time.sleep(0.5)` — the function does almost no computation
    but takes 0.5 seconds of wall-clock time per call.
    """)
    return


@app.cell
def _(simulate_io_task, time):
    _num_tasks = 10
    print(f"Running {_num_tasks} IO-bound tasks sequentially...")
    print()

    _start = time.perf_counter()
    for _i in range(_num_tasks):
        _result = simulate_io_task(task_id=_i)
        print(f"  {_result}")
    io_sequential_time = time.perf_counter() - _start

    print()
    print(f"Sequential time: {io_sequential_time:.2f}s")
    return (io_sequential_time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ThreadPoolExecutor for IO-Bound Tasks

    `concurrent.futures.ThreadPoolExecutor` manages a pool of worker threads. The key API:

    ```python
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(function, iterable)
    ```

    **Why threads work here:** The GIL is released during `time.sleep()` (and real IO operations),
    so multiple threads can sleep concurrently. With 5 workers handling 10 tasks that each take
    0.5s, we expect ~1s total (2 batches of 5).
    """)
    return


@app.cell
def _(concurrent_futures, io_sequential_time, simulate_io_task, time):
    _num_tasks = 10
    _num_workers = 5
    print(f"Running {_num_tasks} IO-bound tasks with {_num_workers} threads...")
    print()

    _start = time.perf_counter()
    with concurrent_futures.ThreadPoolExecutor(max_workers=_num_workers) as _executor:
        _results = list(
            _executor.map(
                simulate_io_task,
                range(_num_tasks),
            )
        )
    io_threaded_time = time.perf_counter() - _start

    for _r in _results:
        print(f"  {_r}")
    print()
    print(f"Threaded time:   {io_threaded_time:.2f}s")
    print(f"Sequential time: {io_sequential_time:.2f}s")
    print(f"Speedup:         {io_sequential_time / io_threaded_time:.1f}x")
    return (io_threaded_time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CPU-Bound Tasks

    A task is **CPU-bound** when it spends most of its time doing computation: number crunching,
    model training, data transformation, etc.

    We use **trial-division primality testing** on large numbers (~2 billion) as our CPU-bound
    workload. Each call requires millions of iterations of pure Python arithmetic — exactly the
    kind of work the GIL serializes.
    """)
    return


@app.cell
def _(CPU_BOUND_NUMBERS, is_prime, time):
    cpu_bound_numbers = CPU_BOUND_NUMBERS

    print(f"Testing {len(cpu_bound_numbers)} numbers for primality sequentially...")
    print()

    _start = time.perf_counter()
    for _n in cpu_bound_numbers:
        _result = is_prime(n=_n)
        _label = "prime" if _result else "composite"
        print(f"  {_n:>15,} → {_label}")
    cpu_sequential_time = time.perf_counter() - _start

    print()
    print(f"Sequential time: {cpu_sequential_time:.2f}s")
    return cpu_bound_numbers, cpu_sequential_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Threads Don't Help CPU-Bound Work

    Because of the GIL, threading a CPU-bound Python workload provides **no speedup**. The OS
    switches between threads, but only one thread executes Python bytecode at any instant.
    The overhead of thread switching can even make it slightly *slower*.

    Let's verify this empirically.
    """)
    return


@app.cell
def _(
    concurrent_futures,
    cpu_bound_numbers,
    cpu_sequential_time,
    is_prime,
    time,
):
    _num_workers = 4
    print(
        f"Running {len(cpu_bound_numbers)} primality tests "
        f"with {_num_workers} threads..."
    )

    _start = time.perf_counter()
    with concurrent_futures.ThreadPoolExecutor(max_workers=_num_workers) as _executor:
        _results = list(
            _executor.map(
                is_prime,
                cpu_bound_numbers,
            )
        )
    cpu_threaded_time = time.perf_counter() - _start

    print()
    for _n, _r in zip(cpu_bound_numbers, _results):
        _label = "prime" if _r else "composite"
        print(f"  {_n:>15,} → {_label}")
    print()
    print(f"Threaded time:   {cpu_threaded_time:.2f}s")
    print(f"Sequential time: {cpu_sequential_time:.2f}s")
    print(f"Speedup:         {cpu_sequential_time / cpu_threaded_time:.1f}x")
    print()
    print("As expected, threads provide ~1x speedup (no improvement) for CPU-bound work.")
    return (cpu_threaded_time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ProcessPoolExecutor for CPU-Bound Work

    `ProcessPoolExecutor` spawns **separate Python processes**, each with its own interpreter
    and its own GIL. This enables true parallelism for CPU-bound tasks.

    The API is nearly identical to `ThreadPoolExecutor` — just swap the class name:

    ```python
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(function, iterable)
    ```

    **Trade-offs vs. threads:**
    - Startup cost: spawning a process is heavier than spawning a thread
    - Data must be **serialized (pickled)** to send between processes
    - No shared memory (each process has its own copy of data)

    These costs are worth it when the computation per task is large enough to amortize the
    overhead.
    """)
    return


@app.cell
def _(
    concurrent_futures,
    cpu_bound_numbers,
    cpu_sequential_time,
    is_prime,
    time,
):
    _num_workers = 4
    print(
        f"Running {len(cpu_bound_numbers)} primality tests "
        f"with {_num_workers} processes..."
    )

    _start = time.perf_counter()
    with concurrent_futures.ProcessPoolExecutor(max_workers=_num_workers) as _executor:
        _results = list(
            _executor.map(
                is_prime,
                cpu_bound_numbers,
            )
        )
    cpu_process_time = time.perf_counter() - _start

    print()
    for _n, _r in zip(cpu_bound_numbers, _results):
        _label = "prime" if _r else "composite"
        print(f"  {_n:>15,} → {_label}")
    print()
    print(f"Process time:    {cpu_process_time:.2f}s")
    print(f"Sequential time: {cpu_sequential_time:.2f}s")
    print(f"Speedup:         {cpu_sequential_time / cpu_process_time:.1f}x")
    return (cpu_process_time,)


@app.cell
def _(concurrent_futures, io_sequential_time, simulate_io_task, time):
    _num_tasks = 10
    _num_workers = 5
    print(f"Running {_num_tasks} IO-bound tasks with {_num_workers} processes...")
    print()

    _start = time.perf_counter()
    with concurrent_futures.ProcessPoolExecutor(max_workers=_num_workers) as _executor:
        _results = list(
            _executor.map(
                simulate_io_task,
                range(_num_tasks),
            )
        )
    io_process_time = time.perf_counter() - _start

    for _r in _results:
        print(f"  {_r}")
    print()
    print(f"Process time:    {io_process_time:.2f}s")
    print(f"Sequential time: {io_sequential_time:.2f}s")
    print(f"Speedup:         {io_sequential_time / io_process_time:.1f}x")
    print()
    print(
        "Processes also speed up IO-bound tasks, but threads are preferred "
        "due to lower overhead."
    )
    return (io_process_time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary Comparison

    Let's compare all approaches side by side. The table and chart below show execution times
    and speedups for each combination of task type and execution method.
    """)
    return


@app.cell
def _(
    cpu_process_time,
    cpu_sequential_time,
    cpu_threaded_time,
    io_process_time,
    io_sequential_time,
    io_threaded_time,
    pl,
):
    summary_df = pl.DataFrame(
        {
            "Task Type": [
                "IO-Bound",
                "IO-Bound",
                "IO-Bound",
                "CPU-Bound",
                "CPU-Bound",
                "CPU-Bound",
            ],
            "Method": [
                "Sequential",
                "Threads",
                "Processes",
                "Sequential",
                "Threads",
                "Processes",
            ],
            "Time (s)": [
                round(io_sequential_time, 2),
                round(io_threaded_time, 2),
                round(io_process_time, 2),
                round(cpu_sequential_time, 2),
                round(cpu_threaded_time, 2),
                round(cpu_process_time, 2),
            ],
            "Speedup": [
                1.0,
                round(io_sequential_time / io_threaded_time, 1),
                round(io_sequential_time / io_process_time, 1),
                1.0,
                round(cpu_sequential_time / cpu_threaded_time, 1),
                round(cpu_sequential_time / cpu_process_time, 1),
            ],
        }
    )
    summary_df
    return (summary_df,)


@app.cell
def _(pl, plt, sns, summary_df):
    _fig, (_ax1, _ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    _io_data = summary_df.filter(
        pl.col("Task Type") == "IO-Bound"
    ).to_pandas()
    _cpu_data = summary_df.filter(
        pl.col("Task Type") == "CPU-Bound"
    ).to_pandas()

    sns.barplot(
        data=_io_data,
        x="Method",
        y="Time (s)",
        hue="Method",
        ax=_ax1,
        edgecolor="k",
        legend=False,
    )
    _ax1.set_title("IO-Bound Tasks")
    _ax1.set_ylabel("Time (s)")
    _ax1.set_xlabel("")

    for _i, (_time, _speedup) in enumerate(
        zip(_io_data["Time (s)"], _io_data["Speedup"])
    ):
        _ax1.text(
            _i,
            _time + 0.1,
            f"{_speedup:.1f}x",
            ha="center",
            fontweight="bold",
        )

    sns.barplot(
        data=_cpu_data,
        x="Method",
        y="Time (s)",
        hue="Method",
        ax=_ax2,
        edgecolor="k",
        legend=False,
    )
    _ax2.set_title("CPU-Bound Tasks")
    _ax2.set_ylabel("Time (s)")
    _ax2.set_xlabel("")

    for _i, (_time, _speedup) in enumerate(
        zip(_cpu_data["Time (s)"], _cpu_data["Speedup"])
    ):
        _ax2.text(
            _i,
            _time + 0.1,
            f"{_speedup:.1f}x",
            ha="center",
            fontweight="bold",
        )

    _fig.suptitle(
        "Execution Time by Task Type and Method",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    | Task Type | Best Approach | Why |
    |-----------|--------------|-----|
    | **IO-bound** (API calls, file IO, DB queries) | `ThreadPoolExecutor` | GIL released during IO; threads are lightweight |
    | **CPU-bound** (pure Python computation) | `ProcessPoolExecutor` | Separate processes bypass the GIL |

    ### Important Nuances

    - **NumPy, pandas, and scikit-learn** release the GIL internally for many operations (they
      call into C/Fortran code). Threads *can* help for these libraries even though the work is
      CPU-bound.
    - **scikit-learn's `n_jobs` parameter** uses `multiprocessing` (via joblib) under the hood.
      Setting `n_jobs=-1` uses all available cores — now you know the mechanism behind it.
    - **Start simple.** Write sequential code first, profile to find bottlenecks, then add
      concurrency only where it helps. Premature concurrency adds complexity without guaranteed
      benefit.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
