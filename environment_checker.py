import sys
import os
import platform
import subprocess
import streamlit as st

st.title("Environment Inspector")

st.header("1. Python Environment")
st.write("Python Version:", sys.version)

st.header("2. System Details")
st.write("OS:", platform.platform())

st.header("3. Installed Packages")
result = subprocess.run(["pip", "list"], capture_output=True, text=True)
st.code(result.stdout)

st.header("4. Streamlit Details")
st.write("Streamlit Version:", streamlit.__version__)
st.write("In Virtual Environment?", sys.prefix != sys.base_prefix)

st.header("5. Environment Variables")
st.json(dict(os.environ))