
## Section 1: Dataset & University Configuration
- **Total Courses Configured:** 65
- **Total Rooms Configured:** 20
- **Total Time Slots Configured:** 50

---

## Section 2: Local Verification & Implemented Solvers
*Check the algorithms and features you successfully ran and verified by placing an `x` in the brackets (e.g., `[x]`):*
- [x] DataLoader & CSV Parsing
- [x] Hard Constraint Validation Engine
- [x] Soft Penalty & Quality Evaluation Engine
- [x] CSP Backtracking with MRV + LCV + Forward Checking
- [x] Min-Conflicts CSP Local Search
- [x] Simulated Annealing / Genetic Algorithm Optimizer
- [x] Timetable CSV Export (<studentid>_timetable.csv)

---

## Section 3: Video Presentation & Project Structure
- **Video Presentation Link:** https://drive.google.com/file/d/1SkgzHpn3uh6MikSojEQ-BClOaaagz06Y/view?usp=drive_link
- **Project Directory Structure:** app.py: Main entry point that runs the scheduling process that call scheduler_core, scheduler_solver and scheduler_penalty file. scheduler_core.py: Loads CSV data and builds CSP domains for 20 courses. scheduler_solver.py: Finds a schedule using CSP search and heuristics. scheduler_penalty.py: Evaluates and improves the schedule using soft constraints and simulated annealing. data/: Contains input datasets and the final timetable CSV.
