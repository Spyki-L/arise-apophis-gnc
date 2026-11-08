#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MiniSim: tiny Basilisk runner without SimulationBaseClass

"""
basilik_minisim.py

Purpose
-------
A minimal, self-contained replacement for `Basilisk.utilities.SimulationBaseClass`
when you just need to wire up processes, tasks, and models — especially for
stand-alone scripts and experiments — without pulling in the full convenience
wrapper.

This module provides a `MiniSim` class that wraps Basilisk's core
`sim_model.SimModel` and exposes a familiar subset of the SimulationBaseClass
API:

- SetProgressBar(on/off)
- CreateNewProcess(name, priority)
- CreateNewTask(name, rate_ns, FirstStart=0)
- AddModelToTask(task_name, model, [config_or_priority], [model_priority])
- InitializeSimulation()
- ConfigureStopTime(stop_time_ns)
- ExecuteSimulation()

Key behavior
------------
1. Core kernel
   - Internally owns a `sim_model.SimModel` instance (`TotalSim`), plus a
     `BSKLogger` used to attach a logger to added models and optional CONFIG
     structs.
   - Processes are created via `simulationArchTypes.ProcessBaseClass` and
     registered using `TotalSim.addNewProcess(proc.processData)`.

2. Processes and tasks
   - `CreateNewProcess(name, priority)`:
       * Returns a `ProcessBaseClass` the caller can later use with
         `proc.addTask(task)`.
       * The process is registered immediately with the underlying SimModel.
   - `CreateNewTask(task_name, task_rate_ns, FirstStart=0)`:
       * Returns a `TaskBaseClass` configured with the given rate and optional
         first-start delay.
       * Tasks are stored internally in `_tasks_by_name` so that
         `AddModelToTask` can look them up by name.

3. Adding models to tasks
   - `AddModelToTask(task_name, model, config_or_priority=None, model_priority=-1)`:
       * Supports the same call styles as SimulationBaseClass:
           - AddModelToTask(name, model)
           - AddModelToTask(name, model, priority:int)
           - AddModelToTask(name, model, CONFIG)
           - AddModelToTask(name, model, CONFIG, priority:int)
       * For C-wrapped modules, you pass the CONFIG struct as
         `config_or_priority`; it will be:
           - Given a `.bskLogger` attribute when possible.
           - Appended to the task’s `TaskModels` list so the SimModel knows
             about it.
       * Python-only models can be added by passing just the model and an
         optional integer priority.

4. Lifecycle
   - `InitializeSimulation()`:
       * Calls `TotalSim.assignRemainingProcs()` to ensure processes are
         mapped to threads.
       * Runs the same init sequence as SimulationBaseClass:
             ResetSimulation()
             selfInitSimulation()
             resetInitSimulation()
   - `ConfigureStopTime(time_stop_ns)`:
       * Stores the final stop time in nanoseconds.
   - `ExecuteSimulation()`:
       * Requires `InitializeSimulation()` to have been called first.
       * Steps the kernel using:
             while TotalSim.NextTaskTime <= stop_time:
                 StepUntilStop(next_stop, -1)
       * Integrates an optional progress bar:
           - Tries `SimulationProgressBar` from
             `Basilisk.utilities.simulationProgressBar`.
           - Falls back to the historical spelling
             `simulationProgessBar`.
           - If neither import is available, uses a no-op stub that satisfies
             the interface.

5. Progress bar
   - `SetProgressBar(True/False)` toggles whether a progress bar is shown in
     `ExecuteSimulation()`.
   - When enabled and available, the progress bar is updated with the current
     kernel time (`NextTaskTime`) and marked complete at the end.

Usage example
-------------
Typical pattern inside a stand-alone Basilisk script:

    from sim_core.basilik_minisim import MiniSim
    from Basilisk.utilities import macros

    sim = MiniSim()
    sim.SetProgressBar(True)

    proc = sim.CreateNewProcess("DynProc")
    dynTask = sim.CreateNewTask("DynTask", macros.sec2nano(0.1))
    proc.addTask(dynTask)

    # Add your models
    sc = spacecraft.Spacecraft()
    sim.AddModelToTask("DynTask", sc)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(600.0))
    sim.ExecuteSimulation()

MiniSim is intentionally minimal: it does not manage logging directories,
scenario configuration, or other high-level features. It’s a light-weight
scheduling kernel for small Basilisk experiments and custom runners.
"""


from Basilisk.architecture import sim_model, bskLogging
from Basilisk.utilities import simulationArchTypes

# Progress bar: support both spellings, and fall back to a no-op if not available
try:
    from Basilisk.utilities.simulationProgressBar import SimulationProgressBar  # preferred
except Exception:
    try:
        # historical module name in some trees
        from Basilisk.utilities.simulationProgessBar import SimulationProgressBar  # noqa: F401
    except Exception:
        class SimulationProgressBar:  # type: ignore
            def __init__(self, *_args, **_kwargs): pass
            def update(self, *_a, **_k): pass
            def markComplete(self): pass
            def close(self): pass


class MiniSim:
    """
    Minimal scheduler to replace Basilisk.utilities.SimulationBaseClass for:
      - SetProgressBar
      - CreateNewProcess / CreateNewTask
      - AddModelToTask
      - InitializeSimulation
      - ConfigureStopTime
      - ExecuteSimulation

    Notes:
    - Uses Basilisk's core SimModel (sim_model.SimModel) directly,
      avoiding the SimulationBaseClass convenience wrapper.
    - Supports adding both Python models and C-wrapped models (via CONFIG structs).
    """
    def __init__(self):
        self.TotalSim = sim_model.SimModel()
        self.bskLogger = bskLogging.BSKLogger()
        self._proc_list = []
        self._tasks_by_name = {}
        self._stop_time = 0
        self._show_progress = False
        self._initialized = False

    # ------------- UX helpers -------------
    def SetProgressBar(self, value: bool):
        self._show_progress = bool(value)

    # ------------- Build graph -------------
    def CreateNewProcess(self, name: str, priority: int = -1):
        """
        Returns a ProcessBaseClass and registers it with the SimModel.
        Caller should later do: proc.addTask(task)
        """
        proc = simulationArchTypes.ProcessBaseClass(name, priority)
        self._proc_list.append(proc)
        # Register process with kernel
        self.TotalSim.addNewProcess(proc.processData)
        return proc

    def CreateNewTask(self, task_name: str, task_rate_ns: int, FirstStart: int = 0):
        """
        Returns a TaskBaseClass configured with rate and optional start delay.
        The caller must add it to a process via proc.addTask(task).
        """
        task = simulationArchTypes.TaskBaseClass(task_name, task_rate_ns, FirstStart)
        self._tasks_by_name[task_name] = task
        return task

    def AddModelToTask(
        self,
        task_name: str,
        model,
        config_or_priority=None,
        model_priority: int = -1
    ):
        """
        Attach `model` to the task named `task_name`.

        Accepted call styles (mimics SimulationBaseClass.AddModelToTask):
          - AddModelToTask(name, model)
          - AddModelToTask(name, model, priority:int)
          - AddModelToTask(name, model, config)
          - AddModelToTask(name, model, config, priority:int)

        For C-wrapped modules, pass the CONFIG struct as `config_or_priority`.
        """
        if task_name not in self._tasks_by_name:
            raise ValueError(f"Task '{task_name}' not found. Create it and add to a process first.")

        cfg = None
        prio = model_priority

        if isinstance(config_or_priority, int):
            prio = config_or_priority
        elif config_or_priority is not None:
            cfg = config_or_priority

        task = self._tasks_by_name[task_name]

        # Try to provide a logger on both the model and (optionally) its CONFIG struct
        for obj in (model, cfg):
            if obj is None:
                continue
            try:
                obj.bskLogger = self.bskLogger
            except Exception:
                pass

        # Register the model with the task
        task.TaskData.AddNewObject(model, prio)

        # Register the CONFIG object if present (CONFIG pattern for C-wrapped modules)
        if cfg is not None:
            task.TaskModels.append(cfg)
        else:
            task.TaskModels.append(model)

    # ------------- Lifecycle -------------
    def InitializeSimulation(self):
        """Wire processes to threads, then run selfInit/reset sequences."""
        # Assign any processes that were not explicitly assigned a thread
        self.TotalSim.assignRemainingProcs()
        # Reset and init sequence (mirrors SimulationBaseClass)
        self.TotalSim.ResetSimulation()
        self.TotalSim.selfInitSimulation()
        self.TotalSim.resetInitSimulation()
        self._initialized = True

    def ConfigureStopTime(self, time_stop_ns: int):
        self._stop_time = int(time_stop_ns)

    def ExecuteSimulation(self):
        """Step the kernel until stop time. Shows a progress bar if enabled."""
        if not self._initialized:
            raise RuntimeError("MiniSim not initialized. Call InitializeSimulation() first.")

        progress = SimulationProgressBar(self._stop_time, self._show_progress)

        # Step until the requested stop time
        while self.TotalSim.NextTaskTime <= self._stop_time:
            # Step to either the next internal event or our final stop time
            next_stop = self._stop_time if self._stop_time >= self.TotalSim.NextTaskTime else self.TotalSim.NextTaskTime
            self.TotalSim.StepUntilStop(next_stop, -1)
            progress.update(self.TotalSim.NextTaskTime)

        progress.markComplete()
        progress.close()
