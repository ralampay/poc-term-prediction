import streamlit as st
import torch
import os
import sys
import json
from lib.multi_layer_perceptron import MultiLayerPerceptron

# CL Only assets
mlp_cl_only_config = "./config/mlp_a_cl_only.json"
mlp_cl_only_model = "./models/model_a_cl.pth"

# CL + HR assets
mlp_cl_hr_config = "./config/mlp_a_cl_hr.json"
mlp_cl_hr_model = "./models/model_a_cl_hr_preterm_term.pth"

# CL + Partus assets
mlp_cl_partus_config = "./config/mlp_a_cl_partus.json"
mlp_cl_partus_model = "./models/model_a_cl_partus_preterm_term.pth"

device = "cpu"

# Load CL Only model
with open(mlp_cl_only_config) as json_file:
    mlp_cfg = json.load(json_file)

model_cl_only = MultiLayerPerceptron(mlp_cfg).to("cpu")
model_cl_only.load_state_dict(torch.load(mlp_cl_only_model))
model_cl_only.eval()

# Load CL + HR model
with open(mlp_cl_hr_config) as json_file:
    mlp_cfg = json.load(json_file)

model_cl_hr = MultiLayerPerceptron(mlp_cfg).to("cpu")
model_cl_hr.load_state_dict(torch.load(mlp_cl_hr_model))
model_cl_hr.eval()

# Load CL + Partus model
with open(mlp_cl_partus_config) as json_file:
    mlp_cfg = json.load(json_file)
model_cl_partus = MultiLayerPerceptron(mlp_cfg).to("cpu")
model_cl_partus.load_state_dict(torch.load(mlp_cl_partus_model))
model_cl_partus.eval()

st.title("Term Prediction")

input_type_cl_only      = "CL Only"
input_type_cl_hr        = "CL + Hardness Ratio"
input_type_cl_partus    = "CL + Partus"

input_type = input_type_cl_only

input_type = st.radio(
    "Select Input Type:",
    (input_type_cl_only, input_type_cl_hr, input_type_cl_partus)
)

if input_type == input_type_cl_only:
    model_classifier = model_cl_only
elif input_type == input_type_cl_hr:
    model_classifier = model_cl_hr
elif input_type == input_type_cl_partus:
    model_classifier = model_cl_partus

# Horizontal line
st.markdown("---")

x1_input = st.checkbox("Advanced Maternal Age")
x2_input = st.checkbox("History of Previous STPB")
x3_input = st.checkbox("History of Miscarriage")
x4_input = st.number_input("Cervical Length")

if input_type == input_type_cl_hr:
    x5_input = st.slider("Hardness Ratio", 0.0, 100.0)

elif input_type == input_type_cl_partus:
    x5_input = st.checkbox("Partus")

if st.button("Predict"):
    input_data = [
        int(x1_input),
        int(x2_input),
        int(x3_input),
        x4_input
    ]

    if input_type == input_type_cl_hr:
        input_data.append(x5_input)

    elif input_type == input_type_cl_partus:
        input_data.append(int(x5_input))

    input_vector = torch.tensor(input_data, dtype=torch.float32)

    print("Input:")
    print(input_vector)

    output = model_classifier(input_vector).detach().cpu().numpy()

    print("Output:")
    print(output)

    prob_preterm = round(output[0] * 100, 2)
    prob_term = round(output[1] * 100, 2)

    # Display the probabilities in two columns
    st.markdown("""---""")
    st.header("Results")
    col1, col2 = st.columns(2)
    col1.metric(label="Probability Preterm", value=f"{prob_preterm:.2f}%")
    col2.metric(label="Probability Term", value=f"{prob_term:.2f}%")
