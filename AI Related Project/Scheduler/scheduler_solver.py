
def scheduling_constraint(course_id, schedule_option, assignment, course_cohort): 
    #check if there is any clash in the room 
    #variable = course id, domain = schedule_option, constrain = room, instructor, cohort, assignmnet = scheduled course
    room, instr, slot = schedule_option
    this_cohort = course_cohort.get(course_id)
    for other_id, other_val in assignment.items():
        other_room, other_instr, other_slot = other_val
        if slot != other_slot:
            continue
        if room == other_room:
            return False          # room clash
        if instr == other_instr:
            return False          # instructor clash
        if this_cohort == course_cohort.get(other_id):
            return False          # cohort clash
    return True # retur if no clash


def heuristic_variable_selection(domains, assignment, course_cohort):
    #find courses that have not been scheduled yet
    unassigned = [cid for cid in domains if cid not in assignment]  #for checking if the course is assigned or not

    def degree(cid): #Degree of Heristic
        this_cohort = course_cohort.get(cid) 
        return sum(1 for other in unassigned
                   if other != cid and course_cohort.get(other) == this_cohort)
        #Go through all unassigned courses and count every course that is not in same year (CSE 342 that belongs to CSE_Y3.)

    # MRV first (smallest domain), degree heuristic as tie-break (most constrained cohort-mates)
    unassigned.sort(key=lambda cid: (len(domains[cid]), -degree(cid)))  #return smallest domain #if tie use schedule option
    return unassigned[0] #return the course with the 

def LCV(course_id, domain, assignment, domains, course_cohort): 
    #Choose the schedule option that restricts the fewest options for other unassigned courses.
    this_cohort = course_cohort.get(course_id) #to check cohort clash

    # to calculates how many conflicts a particular schedule option could cause for other unassigned courses.
    def conflicts_caused(value):
        room, instr, slot = value #separate v value
        count = 0
        for other_id, other_domain in domains.items(): #checking if any other course is 
            if other_id == course_id or other_id in assignment: #skip the assigned course
                continue
            for other_room, other_instr, other_slot in other_domain: #check schedule option of another course
                if slot != other_slot: #skip different time slot
                    continue
                if room == other_room or instr == other_instr or this_cohort == course_cohort.get(other_id): 
                    #if there is any conflit of above conditions meet count as conflit
                    count += 1
        return count
    return sorted(domain, key=conflicts_caused)  #return least constraint schedule option

def forward_check(course_id, value, domains, assignment, course_cohort):
    #remove schedule option values that conflict with the newly assigned course.
    room, instr, slot = value #separate schedule option value
    this_cohort = course_cohort.get(course_id) #current assigned course 
    removed = {}
    for other_id, other_domain in domains.items():
        if other_id == course_id or other_id in assignment: #remove the current schedule option
            continue
        new_domain = [
            v for v in other_domain #v = (room, instructor, slot)
            if not (v[2] == slot and (v[0] == room or v[1] == instr or course_cohort.get(other_id) == this_cohort))
        ]
        if not new_domain:
            return None            # a domain went empty — this branch is dead
        removed[other_id] = new_domain
    return removed #return remaing possible schedule option


def backtrack(assignment, domains, course_cohort):
    #schedule courses one by one and if choice causes a problem, go back and try another choice
    if len(assignment) == len(domains): #check if all cause are assigned
        return dict(assignment)

    course_id = heuristic_variable_selection(domains, assignment, course_cohort) #find the course that is not schedule yet with smallest domain
    for value in LCV(course_id, domains[course_id], assignment, domains, course_cohort): #get ordered least constraint schedule option
        if scheduling_constraint(course_id, value, assignment, course_cohort): #check if there is clash in room,instructor,cohor or not and will have True or False 
            assignment[course_id] = value #if no clash return true temporaty assing as ok
            removed = forward_check(course_id, value, domains, assignment, course_cohort) #remove the confliting schedule option
            if removed is not None: 
                new_domains = dict(domains) 
                new_domains.update(removed) #create an update domain for next recursive function call
                result = backtrack(assignment, new_domains, course_cohort) #to schedule the next remaining course
                if result is not None:
                    return result #if valid result is return
            del assignment[course_id] # if not None, go back and try another schedule option
    return None


def backtracking_search(domains, course_cohort): #the start of the scheduler algorithm
    return backtrack({}, dict(domains), course_cohort) #there is no assign room in the very start


import random

def conflict_count(course_id, value, assignment, course_cohort):
    room, instr, slot = value
    this_cohort = course_cohort.get(course_id)
    count = 0
    for other_id, other_val in assignment.items():
        if other_id == course_id:
            continue
        other_room, other_instr, other_slot = other_val
        if slot != other_slot:
            continue
        if room == other_room or instr == other_instr or this_cohort == course_cohort.get(other_id):
            count += 1
    return count

def min_conflicts(assignment, domains, course_cohort, max_steps=1000):
    current = dict(assignment)
    for step in range(max_steps):
        conflicted = [cid for cid in current if conflict_count(cid, current[cid], current, course_cohort) > 0]
        if not conflicted:
            return current, step   # no conflicts left -> solved (or already was)
        course_id = random.choice(conflicted)
        best_value = min(domains[course_id], key=lambda v: conflict_count(course_id, v, current, course_cohort))
        current[course_id] = best_value
    return current, max_steps
