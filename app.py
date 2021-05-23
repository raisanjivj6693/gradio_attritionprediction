import gradio as gr
import pandas as pd
import numpy as np
import sklearn
import pickle
import numpy as np

model = pickle.load(open('random_forest_classification_model_top.pkl', 'rb'))

def predict_attrition(overtime, Age, TotalWorkingYears, MonthlyIncome, JobSatisfaction, YearsAtCompany, EnvironmentSatisfaction, RelationshipSatisfaction, DistanceFromHome, NumCompaniesWorked):
    if overtime == "Yes":
        overtime = 1
    else:
        overtime = 0
    df = pd.DataFrame.from_dict({'Overtime': [overtime], 'Age': [Age], 'Total Working Years': [TotalWorkingYears], 'Monthly Income': [MonthlyIncome], 'Job Satisfaction': [JobSatisfaction], 'Years At Company': [YearsAtCompany], 'Environment Satisfaction': [EnvironmentSatisfaction], 'Relationship Satisfaction': [RelationshipSatisfaction], 'Distance From Home': [DistanceFromHome], 'Num Companies Worked': [NumCompaniesWorked] })
    # df = encode_sex(df)
    # df = encode_fares(df)
    # df = encode_ages(df)
    pred = model.predict_proba(df)[0]
    return {'No Attrition': pred[0], 'Attrition': pred[1]}

overtime = gr.inputs.Radio(['Yes', 'No'], label="Overtime?")
# if (overtime == 'Yes'):
#     overtime = 1
# else:
#     overtime = 0
Age = gr.inputs.Slider(minimum=18, maximum=60, default=18, step=1, label="Age")
TotalWorkingYears = gr.inputs.Slider(minimum=0, maximum=42, default=0, step=1, label="Total Working Years")
MonthlyIncome = gr.inputs.Slider(minimum=1000, maximum=100000, default=1000, step=100, label="Monthly Income (in $)")
JobSatisfaction = gr.inputs.Slider(minimum=1, maximum=4, default=1, step=1, label="Job Satisfaction")
YearsAtCompany = gr.inputs.Slider(minimum=0, maximum=42, default=0, step=1, label="Years At Company")
EnvironmentSatisfaction = gr.inputs.Slider(minimum=1, maximum=4, step=1, default=1, label="Environment Satisfaction")
RelationshipSatisfaction = gr.inputs.Slider(minimum=1, maximum=4, step=1, default=1, label="Relationship Satisfaction")
DistanceFromHome = gr.inputs.Slider(minimum=1, maximum=30, default=1, step=1, label="Distance From Home")
NumCompaniesWorked = gr.inputs.Slider(minimum=0, maximum=40, default=0, step=1, label="Number of Companies Worked")

#def main():
gr.Interface(fn = predict_attrition, inputs = [overtime, Age, TotalWorkingYears, MonthlyIncome, JobSatisfaction, YearsAtCompany, EnvironmentSatisfaction, RelationshipSatisfaction, DistanceFromHome, NumCompaniesWorked],  "label", capture_session=True).launch()
    
#if __name__ == '__main__':
#	main()
