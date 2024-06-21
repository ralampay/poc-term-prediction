import streamlit as st
import torch
import os
import sys
import json
from lib.multi_layer_perceptron import MultiLayerPerceptron

mlp_cl_only_config = "./config/mlp_a_cl_only.json"
mlp_cl_only_model = "./models/model_a_cl.pth"

device = "cpu"

with open(mlp_cl_only_config) as json_file:
    mlp_cfg = json.load(json_file)

model_cl_only = MultiLayerPerceptron(mlp_cfg).to("cpu")
model_cl_only.load_state_dict(torch.load(mlp_cl_only_model))
model_cl_only.eval()

st.title("Term Prediction")

x1_input = st.checkbox("Advanced Maternal Age")
x2_input = st.checkbox("History of Previous STPB")
x3_input = st.checkbox("History of Miscarriage")
x4_input = st.number_input("Cervical Length")

if st.button("Predict"):
#    st.header("Inputs")
#    st.write(f"Advanced Maternal Age: {'Yes' if x1_input else 'No'}")
#    st.write(f"History of Previous STPB: {'Yes' if x2_input else 'No'}")
#    st.write(f"History of Miscarriage: {'Yes' if x3_input else 'No'}")
#    st.write(f"Age: {x4_input}")

    input_vector = torch.tensor([
        int(x1_input),
        int(x2_input),
        int(x3_input),
        x4_input
    ], dtype=torch.float32)

    print("Input:")
    print(input_vector)

    output = model_cl_only(input_vector).detach().cpu().numpy()

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
