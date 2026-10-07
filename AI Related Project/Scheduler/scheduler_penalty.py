
def score_capacity(assignment, course_cohort, cohort_size, room_id):
#Looks at each assigned course.
#Finds the number of students in its cohort.
#Finds the room's capacity.
#Compares the student number with the room capacity.
#Adds a penalty when the room is too small or significantly oversized.

    penalty = 0
    for course_id, (room, instr, slot) in assignment.items():
        cid = course_cohort.get(course_id)
        size = cohort_size.get(cid)          # cohort's student count
        room_row = room_id.get(room)
        if size is None or room_row is None:
            continue
        capacity = int(room_row["scheduling_capacity"])
        if size > capacity:
            penalty += 1000                   # H2: room too small — hard violation, scored not filtered
        elif (capacity - size) > 30:
            penalty += 5                      # S2: room way oversized — wasted capacity
    return penalty

def score_transit(assignment, course_cohort, slot_info, room_id):
    from collections import defaultdict
    by_cohort_day = defaultdict(list)
    for course_id, (room, instr, slot) in assignment.items():
        cid = course_cohort.get(course_id)
        s = slot_info.get(slot)
        if s is None:
            continue
        by_cohort_day[(cid, s["day"])].append((s["start_time"], room))

    penalty = 0
    for key, sessions in by_cohort_day.items():
        sessions.sort()                       # sort by start_time
        for (t1, room1), (t2, room2) in zip(sessions, sessions[1:]):
            b1 = room_id.get(room1, {}).get("building")
            b2 = room_id.get(room2, {}).get("building")
            if b1 and b2 and b1 != b2:
                penalty += 6                   # S5: back-to-back sessions, different buildings
    return penalty

def slot_order(slot_info):
    by_day = {}
    for sid, s in slot_info.items():
        day = s["day"]
        if day not in by_day:
            by_day[day] = []
        by_day[day].append((s["start_time"], sid))

    order = {}
    for day in by_day:
        slots = sorted(by_day[day])
        for idx in range(len(slots)):
            sid = slots[idx][1]
            order[sid] = idx
    return order

def score_gaps(assignment, course_cohort, slot_info):
    order = slot_order(slot_info)
    groups = {}
    for course_id in assignment:
        room, instr, slot = assignment[course_id]
        cid = course_cohort.get(course_id)
        s = slot_info.get(slot)
        if s is None:
            continue
        key = (cid, s["day"])
        if key not in groups:
            groups[key] = []
        groups[key].append(order[slot])

    penalty = 0
    for key in groups:
        indices = sorted(groups[key])
        for i in range(len(indices) - 1):
            gap = indices[i+1] - indices[i]
            if gap == 2:          # exactly one slot skipped
                penalty += 8
    return penalty

def count_fatigue(groups):
    penalty = 0
    for key in groups:
        indices = sorted(set(groups[key]))
        run = 1
        for i in range(len(indices) - 1):
            if indices[i+1] - indices[i] == 1:
                run = run + 1
            else:
                if run > 2:
                    penalty += 12
                run = 1
        if run > 2:
            penalty += 12
    return penalty

def score_fatigue(assignment, course_cohort, slot_info):
    order = slot_order(slot_info)
    cohort_groups = {}
    instr_groups = {}
    for course_id in assignment:
        room, instr, slot = assignment[course_id]
        cid = course_cohort.get(course_id)
        s = slot_info.get(slot)
        if s is None:
            continue
        ckey = (cid, s["day"])
        ikey = (instr, s["day"])
        if ckey not in cohort_groups:
            cohort_groups[ckey] = []
        cohort_groups[ckey].append(order[slot])
        if ikey not in instr_groups:
            instr_groups[ikey] = []
        instr_groups[ikey].append(order[slot])

    return count_fatigue(cohort_groups) + count_fatigue(instr_groups)

import random
import scheduler_solver

def total_penalty(assignment, course_cohort, cohort_size, room_id, slot_info):
    return (score_capacity(assignment, course_cohort, cohort_size, room_id)
          + score_transit(assignment, course_cohort, slot_info, room_id)
          + score_gaps(assignment, course_cohort, slot_info)
          + score_fatigue(assignment, course_cohort, slot_info))

def simulated_annealing(assignment, domains, course_cohort, cohort_size, room_id, slot_info,
                         iterations=1000, start_temp=100.0, cooling=0.995):
    current = dict(assignment)
    current_score = total_penalty(current, course_cohort, cohort_size, room_id, slot_info)
    temp = start_temp

    for step in range(iterations):
        course_id = random.choice(list(current.keys()))
        new_value = random.choice(domains[course_id])

        others = {k: v for k, v in current.items() if k != course_id}
        if not scheduler_solver.scheduling_constraint(course_id, new_value, others, course_cohort):
            temp = temp * cooling
            continue   # would break a hard constraint (room/instructor/cohort clash) — reject outright

        candidate = dict(current)
        candidate[course_id] = new_value
        candidate_score = total_penalty(candidate, course_cohort, cohort_size, room_id, slot_info)

        delta = candidate_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / temp):
            current = candidate
            current_score = candidate_score

        temp = temp * cooling

    return current, current_score