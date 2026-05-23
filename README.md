# Multi-Agent Pickup and Dropoff for Makespan Optimization

A planning-and-control stack for a fleet of differential-drive robots that must collect scattered items and deliver them to designated collection zones in a warehouse-like map. The system uses a one-time global planning phase to coordinate the team with goal assignment and high-level routes, then runs decentralized execution: each robot acts on its local LiDAR-based observations and treats other robots as dynamic obstacles. The objective is to minimize the overall completion time while avoiding collisions and operating within real-time control constraints.

<table>
  <tr>
    <td style="vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/df969461-d613-4f6d-8626-d2999ebb63d8" width="100%">
    </td>
    <td style="vertical-align: top;">
      <img width="600" height="450" src="https://github.com/user-attachments/assets/75e6d797-3746-4c64-a829-eccc8d27a054" width="100%">
    </td>
  </tr>
</table>

This repository has been trimmed down to the implemented PDM4AR exercise 14 solution: multi-agent collection. Older exercise stubs, test definitions, notebooks, generated graph dumps, and course-site pages for exercises 1-13 have been removed.

## Where things live

- `src/pdm4ar/exercises/ex14/`: the submitted agent and helper code.
- `src/pdm4ar/exercises_def/ex14/`: the local exercise runner, metrics, scenario configs, isolated agent process wrapper, and random config generator.
- `src/pdm4ar/exercises_def/structures*.py`: shared evaluator scaffolding still needed by the runner.
- `src/pdm4ar/available_exercises.py`: the exercise registry, now containing only exercise `"14"`.
- `docs/14-multiagent_collection.md`: the retained exercise statement, with its local images under `docs/img/`.

## Running exercise 14

Install dependencies:

```bash
poetry install
```

Run the local scenarios:

```bash
poetry run pdm4ar-exercise --exercise 14
```

or:

```bash
make run-ex14
```

The report is written under `out/14/index.html`.

## Scenario configs

The default local scenarios are listed in `src/pdm4ar/exercises_def/ex14/ex14.py` and stored as:

- `config_1.yaml`
- `config_2.yaml`
- `config_3.yaml`

For debugging, `config_debug.yaml` is available but is not included in the default run. New generated configs can be created with `src/pdm4ar/exercises_def/ex14/random_config.py` and then added to the config list in `ex14.py`.
