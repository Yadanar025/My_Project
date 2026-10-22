
## Section 1: Dataset & University Configuration
- **Total Courses Configured:** 65
- **Total Rooms Configured:** 20
- **Total Time Slots Configured:** 50

---

## Section 2: Project Structure
- **Project Directory Structure:** app.py: Main entry point that runs the scheduling process that call scheduler_core, scheduler_solver and scheduler_penalty file. scheduler_core.py: Loads CSV data and builds CSP domains for 20 courses. scheduler_solver.py: Finds a schedule using CSP search and heuristics. scheduler_penalty.py: Evaluates and improves the schedule using soft constraints and simulated annealing. data/: Contains input datasets and the final timetable CSV.
