#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# In[2]:


#  Load Data for drop out rate
@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_drop_out.csv")
    return data

drop_out = load_data()

# Define  (X) and  (y)
X = drop_out.drop(columns="Drop_Out")
y = drop_out["Drop_Out"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y, random_state=42)

#SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train_resampled)
X_test_scale = scaler.transform(X_test)

#random forest
rf = RandomForestClassifier(n_estimators = 10,random_state=42)
rf.fit(X_train_scale,y_train_resampled)

#confusion matrix
y_pred_dt=rf.predict(X_test_scale)
conf_matrix = confusion_matrix(y_test,y_pred_dt)

# In[3]:


#  Load Data for att rate
@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_attendance_rate.csv")
    return data

attendance_rate = load_data()

# Define  (X) and  (y)
X_att = attendance_rate.drop(columns="Rate_%")
y_att = attendance_rate["Rate_%"]

# Train-Test Split
X_att_train, X_att_test, y_att_train, y_att_test = train_test_split(
    X_att, y_att, test_size=0.3, random_state=42)

# Train Random Forest Model for Att Rate
# rf_att = RandomForestRegressor(n_estimators=5,max_depth=1,min_samples_leaf=2,)
# rf_att.fit(X_att_train,y_att_train)

scaler = StandardScaler()
X_att_scaled = scaler.fit_transform(X_att)
X_test_att_scaled = scaler.transform(X_att_test)
lasso_att_lr_best = Lasso(alpha=0.001).fit(X_att_scaled, y_att)



# Streamlit UI
st.title("Attendance Rate Prediction Dashboard")
st.sidebar.header("📝 Input Features")



# In[6]:


# Sidebar Inputs
age = st.sidebar.slider("Age", 17, 50, 25)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
expected_learning_hour = st.sidebar.selectbox("Expected Learning Hour", ["above 5 hour", "3 to 5 hour", "2 to 3 hour", "2 hour"])
current_situation = st.sidebar.selectbox("Current Situation", ["Student", "Unemployed", "Employed", "None of the above"])
country = st.sidebar.selectbox("Country", ["Myanmar-Yangon", "Myanmar-Mandalay", "Myanmar-other cities", "Other Countries"])
pre_knowledge = st.sidebar.number_input("Pre-Knowledge Level (0-5)", 0, 5, 2)
internet_type = st.sidebar.radio("Type of Internet", ["Home Wi-Fi", "Others"])
device_used = st.sidebar.radio("Device Used", ["Own Computer", "Not Own"])
academic_status = st.sidebar.radio("Academic Status", ["Graduate (Degree holder)", "Undergraduate Student", "Other"])
interest_course = st.sidebar.selectbox("Interest Course", ["Data Scientist using Python", "Data Analytics using Python", "Associate Data Analyst in SQL", "Data Analytics using R", "Tableau", "Other"])

#  User Input to Model-Compatible Format
input_data_att_rate = {
    # Pre-Knowledge
    "Pre_Knowledge_Data": pre_knowledge,
    
    # Convert Expected Learning Hour to numeric value
    "Learning_hour": 6 if expected_learning_hour == "above 5 hour" else (4 if expected_learning_hour == "3 to 5 hour" else (2.5 if expected_learning_hour == "2 to 3 hour" else 2)),
    
    # Time_duration, Discussion, Presentation, Assignment, Participation (mean values or user input can be adjusted here)
    "Time_duration" :  attendance_rate["Time_duration"].mean(),
    "discussion_rate": attendance_rate["discussion_rate"].mean(),
    "presentation_project_rate": attendance_rate["presentation_project_rate"].mean(),
    "assignment_rate": attendance_rate["assignment_rate"].mean(),
    "participation": attendance_rate["participation"].mean(),
    
    # Gender One-Hot 
    "Gender_Female": 1 if gender == "Female" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,

    # Current Situation One-Hot 
    "Current_Situation_Employed": 1 if current_situation == "Employed" else 0,
    "Current_Situation_None of the above": 1 if current_situation == "None of the above" else 0,
    "Current_Situation_Student": 1 if current_situation == "Student" else 0,
    "Current_Situation_Unemployed": 1 if current_situation == "Unemployed" else 0,

    # Internet Type One-Hot 
    "Type_of_Internet_Home Wi-Fi": 1 if internet_type == "Home Wi-Fi" else 0,
    "Type_of_Internet_Others": 1 if internet_type == "Others" else 0,

    # Device Used One-Hot 
    "Device_used_Not Own": 1 if device_used == "Not Own" else 0,
    "Device_used_Own Computer": 1 if device_used == "Own Computer" else 0,

    # Academic Status One-Hot 
    "Academic_career_Graduate (Degree holder)": 1 if academic_status == "Graduate (Degree holder)" else 0,
    "Academic_career_Other": 1 if academic_status == "Other" else 0,
    "Academic_career_Undergraduate Student": 1 if academic_status == "Undergraduate Student" else 0,
    
    # Course Interest One-Hot 
    "Course_Wish_Join_Associate Data Analyst in SQL": 1 if interest_course == "Associate Data Analyst in SQL" else 0,
    "Course_Wish_Join_Data Analytics using Python": 1 if interest_course == "Data Analytics using Python" else 0, 
    "Course_Wish_Join_Data Analytics using R": 1 if interest_course == "Data Analytics using R" else 0,
    "Course_Wish_Join_Data Scientist using Python": 1 if interest_course == "Data Scientist using Python" else 0,
    "Course_Wish_Join_Other": 1 if interest_course == "Other" else 0,
    "Course_Wish_Join_Tableau": 1 if interest_course == "Tableau" else 0,

    # Age Mapping One-Hot 
    #**{f"age_{i}.0": 1 if age == i else 0 for i in range(22, 50)},  # Generate columns for age 22.0 to 49.0
    "age_22.0": 1 if age==22 else 0,
    "age_23.0": 1 if age==23 else 0,
    "age_24.0": 1 if age==24 else 0,
    "age_25.0": 1 if age==25 else 0,
    "age_26.0": 1 if age==26 else 0,
    "age_27.0": 1 if age==27 else 0,
    "age_28.0": 1 if age==28 else 0,
    "age_29.0": 1 if age==29 else 0,
    "age_30.0": 1 if age==30 else 0,
    "age_31.0": 1 if age==31 else 0,
    "age_32.0": 1 if age==32 else 0,
    "age_33.0": 1 if age==33 else 0,
    "age_34.0": 1 if age==34 else 0,
    "age_35.0": 1 if age==35 else 0,
    "age_36.0": 1 if age==36 else 0,
    "age_37.0": 1 if age==37 else 0,
    "age_38.0": 1 if age==37 else 0,
    "age_39.0": 1 if age==37 else 0,
    "age_41.0": 1 if age==41 else 0,
    "age_42.0": 1 if age==42 else 0,    
    "age_43.0": 1 if age==43 else 0,
    "age_44.0": 1 if age==44 else 0,
    "age_45.0": 1 if age==45 else 0,
    "age_46.0": 1 if age==46 else 0,
    "age_48.0": 1 if age==48 else 0,
    
    # Urbanization: Rural or Urban One-Hot
    "Urbanization_Rural": 1 if country == "Myanmar-other cities" else 0,
    "Urbanization_Urban": 1 if country in ["Myanmar-Yangon", "Myanmar-Mandalay", "Other Countries"] else 0
}

# Convert Input to DataFrame
input_df = pd.DataFrame([input_data_att_rate])

# Reorder Input Data Columns to Match Model's Expected Feature Order
input_df = input_df[X_att.columns]


# # User Input data
# st.write("### User Input Data")
# st.dataframe(input_df)


# In[7]:


input_data_drop_out = {
    # Pre-Knowledge
    "Pre_Knowledge_Data": pre_knowledge,
    
    # Convert Expected Learning Hour to numeric value
    "Learning_hour": 6 if expected_learning_hour == "above 5 hour" else (4 if expected_learning_hour == "3 to 5 hour" else (2.5 if expected_learning_hour == "2 to 3 hour" else 2)),
    
    # Time_duration, Discussion, Presentation, Assignment, Participation (mean values or user input can be adjusted here)
    "Time_duration" :  attendance_rate["Time_duration"].mean(),
    "discussion_rate": attendance_rate["discussion_rate"].mean(),
    "presentation_project_rate": attendance_rate["presentation_project_rate"].mean(),
    "assignment_rate": attendance_rate["assignment_rate"].mean(),
    "participation": attendance_rate["participation"].mean(),
    
    # Gender One-Hot 
    "Gender_Female": 1 if gender == "Female" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,

    # Current Situation One-Hot 
    "Current_Situation_Employed": 1 if current_situation == "Employed" else 0,
    "Current_Situation_None of the above": 1 if current_situation == "None of the above" else 0,
    "Current_Situation_Student": 1 if current_situation == "Student" else 0,
    "Current_Situation_Unemployed": 1 if current_situation == "Unemployed" else 0,

    # Internet Type One-Hot 
    "Type_of_Internet_Home Wi-Fi": 1 if internet_type == "Home Wi-Fi" else 0,
    "Type_of_Internet_Others": 1 if internet_type == "Others" else 0,

    # Device Used One-Hot 
    "Device_used_Not Own": 1 if device_used == "Not Own" else 0,
    "Device_used_Own Computer": 1 if device_used == "Own Computer" else 0,

    # Academic Status One-Hot 
    "Academic_career_Graduate (Degree holder)": 1 if academic_status == "Graduate (Degree holder)" else 0,
    "Academic_career_Other": 1 if academic_status == "Other" else 0,
    "Academic_career_Undergraduate Student": 1 if academic_status == "Undergraduate Student" else 0,
    
    # Course Interest One-Hot 
    "Course_Wish_Join_Associate Data Analyst in SQL": 1 if interest_course == "Associate Data Analyst in SQL" else 0,
    "Course_Wish_Join_Data Analytics using Python": 1 if interest_course == "Data Analytics using Python" else 0, 
    "Course_Wish_Join_Data Analytics using R": 1 if interest_course == "Data Analytics using R" else 0,
    "Course_Wish_Join_Data Scientist using Python": 1 if interest_course == "Data Scientist using Python" else 0,
    "Course_Wish_Join_Other": 1 if interest_course == "Other" else 0,
    "Course_Wish_Join_Tableau": 1 if interest_course == "Tableau" else 0,

    # Age Mapping One-Hot 
    #**{f"age_{i}.0": 1 if age == i else 0 for i in range(22, 50)},  # Generate columns for age 22.0 to 49.0
    "age_22.0": 1 if age==22 else 0,
    "age_23.0": 1 if age==23 else 0,
    "age_24.0": 1 if age==24 else 0,
    "age_25.0": 1 if age==25 else 0,
    "age_26.0": 1 if age==26 else 0,
    "age_27.0": 1 if age==27 else 0,
    "age_28.0": 1 if age==28 else 0,
    "age_29.0": 1 if age==29 else 0,
    "age_30.0": 1 if age==30 else 0,
    "age_31.0": 1 if age==31 else 0,
    "age_32.0": 1 if age==32 else 0,
    "age_33.0": 1 if age==33 else 0,
    "age_34.0": 1 if age==34 else 0,
    "age_35.0": 1 if age==35 else 0,
    "age_36.0": 1 if age==36 else 0,
    "age_37.0": 1 if age==37 else 0,
    "age_38.0": 1 if age==37 else 0,
    "age_39.0": 1 if age==37 else 0,
    "age_41.0": 1 if age==41 else 0,
    "age_42.0": 1 if age==42 else 0,    
    "age_43.0": 1 if age==43 else 0,
    "age_44.0": 1 if age==44 else 0,
    "age_45.0": 1 if age==45 else 0,
    "age_46.0": 1 if age==46 else 0,
    "age_48.0": 1 if age==48 else 0,
    
    # Urbanization: Rural or Urban One-Hot
    "Urbanization_Rural": 1 if country == "Myanmar-other cities" else 0,
    "Urbanization_Urban": 1 if country in ["Myanmar-Yangon", "Myanmar-Mandalay", "Other Countries"] else 0
}

# Convert Input to DataFrame
input_df_dropout = pd.DataFrame([input_data_drop_out])

# Reorder Input Data Columns to Match Model's Expected Feature Order
input_df_dropout = input_df[X_att.columns]

# User Input data
st.write("### 🔍 User Input Data")
st.dataframe(input_df_dropout)
if st.checkbox("📂 Show Data"):
    st.write(attendance_rate.head())


# In[8]:


# 🚀 Predict Attendance Rate and drop out
if st.sidebar.button("🔮 Predict"):
    prediction_dropout = rf.predict(input_df_dropout)[0]
    prediction = "Not Drop Out" if prediction_dropout == 0 else "Drop Out"
    st.subheader("🎯 Predicted Drop Out")
    st.write(f"📌 **{prediction}**")
    st.subheader("Confusion Matrix") 
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
    prediction_att = lasso_att_lr_best.predict(input_df)[0]
    st.subheader("🎯 Predicted Attendance Rate")
    prediction_att = 100 if prediction_att>100 else prediction_att
    st.write(f"📌 **{prediction_att:.2f}%**")
    # 📊 Show Model Performance
    st.subheader("📈 Model Performance")
    st.write(f"📉 **Mean Squared Error:** {mean_squared_error(y_att_test, lasso_att_lr_best.predict(X_test_att_scaled)):.2f}")
    st.write(f"📏 **RMSE:** {mean_squared_error(y_att_test, lasso_att_lr_best.predict(X_test_att_scaled)) ** 0.5:.2f}")
    # 🔍 View Dataset







