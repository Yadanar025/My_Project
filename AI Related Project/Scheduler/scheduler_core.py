import csv
import os

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))
    
def build_domain(data_dir): 
    cohort = load_csv(os.path.join(data_dir,"student_cohorts.csv")) #data input
    courses = load_csv(os.path.join(data_dir,"courses.csv"))
    instructor = load_csv(os.path.join(data_dir,"instructors.csv"))
    rooms = load_csv(os.path.join(data_dir,"rooms.csv"))
    time_slots = load_csv((os.path.join(data_dir,"time_slots.csv")))

    room_id = {i["room_id"]: i for i in rooms}
    slot_info = {i["slot_id"]: i for i in time_slots}   # slot_id -> full row (day, start_time, end_time)
    name_to_id = {i["instructor_name"]: i["instructor_id"] for i in instructor}   # instructor name -> instructor_id
    time_slots = dict(slot_info)
    #cohorts_size = {i["size"]: i for i in cohort}
    #cohorts_mandatory = {i["mandatory_courses"]: i for i in cohort}
    #instructor_course = {i["courses"]: i for i in instructor}
    #instructor_name = {i["instructor_name"]: i for i in instructor}
    #instructor_term = {i["term"]: i for i in instructor}
    courses_id = {i["course_id"]: i for i in courses}
    #course_instructor = {i["instructors"]: i for i in courses}
    #course_roomsmentions = {i["rooms_mentioned"]: i for i in courses}
    #course_term = {i["term"]: i for i in courses}

    # print("room id")
    # print(room_id)
    # print("time_slots")
    # print(time_slots)
    # #print("cohorts_size")
    # #print(cohorts_size)
    # #print("cohorts_mandatory")
    # #print(cohorts_mandatory)
    # #print("instructor_course")
    # #print(instructor_course)
    # #print("instructor_name")
    # #print(instructor_name)
    # #print("instructor_term")
    # #print(instructor_term)
    # print("courses_id")
    # print(courses_id)
    # print("course_instructor")
    # #print(course_instructor)
    # #print("course_instructor")
    # #print(course_instructor)
    # #print("course_roomsmentions")
    # #print(course_roomsmentions)
    # #print("course_term")
    # #print(course_term)

    #for splitting the group room mentions
    def split_room_mention(mentions): 
        mentions = mentions.strip() #for TECH 201/202/203/204
        if "/" not in mentions:
            return [mentions]
        room, number = mentions.rsplit(" ",1)
        return[f"{room} {n}" for n in number.split("/")]
    #print(split_room_mention("TECH 201/202/203/204"))

    #possible room for each course
    def p_room (course_info,cohort_size,room_id):
        groups = course_info["rooms_mentioned"].split(";") #because there can be multiple room groups like "AUD 100; HALL B"
        valid =[]
        for group in groups:
            room_ids = split_room_mention(group) #because there are some room mentions like TECH 201/202/203/204
            for i in room_ids:
                room = room_id.get(i)
                if room is None:
                    continue
                if room["available_for_regular_classes"]!="Yes": #check if it is ok to take for regular class
                    continue
                # if int(room["scheduling_capacity"])<cohort_size:
                #     continue
                valid.append(i)
        return valid #final valid room

    def instructors_name(course_info):
        names = course_info["instructors"].split(";")
        names = [n.strip() for n in names if n.strip()]
        return names if names else ["others"] #return instructor name

    time_slots = [row["slot_id"] for row in time_slots.values()] #return timeslot

    def cohort_size_for(course_id):
        for i in cohort:
            mandatory = [m.strip() for m in i["mandatory_courses"].split(";")] #add mandatory course
            if course_id in mandatory:
                return int (i["size"])
        return None

    from itertools import product
    
    domains = {}
    for course_id, course_row in courses_id.items():
        size = cohort_size_for(course_id)  #get cohort size
        if size is None: #for none mandatory course
            continue
        rooms = p_room(course_row, cohort_size_for(course_id), room_id) #valide room
        instrs = instructors_name(course_row)
        domains[course_id] = list(product(rooms, instrs, time_slots)) #will have room, instructor and slot

    course_cohort = {}
    for c in cohort:
        mandatory = [m.strip() for m in c["mandatory_courses"].split(";")]
        for cid in mandatory:
            course_cohort[cid] = c["cohort_id"]
    cohort_size = {c["cohort_id"]: int(c["size"]) for c in cohort}   # cohort_id -> size, for capacity scoring
    return domains, course_cohort, slot_info, name_to_id, room_id, cohort_size

# for cid, d in domains.items():
#     print(cid, len(d), d[:2])
# print(domains["CSE 101"][0:5])