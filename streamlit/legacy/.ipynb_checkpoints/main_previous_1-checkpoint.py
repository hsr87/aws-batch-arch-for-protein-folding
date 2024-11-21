# Standard library imports
import os
import sys
import json
import time
import signal
import inspect
import threading
from datetime import datetime
from decimal import Decimal
from queue import Queue
from typing import Annotated, Any, Dict, List, Sequence, Tuple, TypedDict, Callable, TypeVar

# Data processing and analysis
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator

# AWS related
import boto3
from botocore.config import Config

# Batch processing and AlphaFold
from batchfold.batchfold_environment import BatchFoldEnvironment
from batchfold.batchfold_target import BatchFoldTarget
from batchfold.jackhmmer_job import JackhmmerJob
from batchfold.alphafold2_job import AlphaFold2Job

# Language models and processing
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import graph, END, StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Bioinformatics
from Bio import SeqIO

# import local modules
from src_streamlit.agent import DrugDiscoveryAssistantAgent
from src_streamlit import bedrock
from src_streamlit.bedrock import bedrock_model, bedrock_info, bedrock_utils


dir_current = os.path.abspath("")
dir_parent = os.path.dirname(dir_current)
if dir_parent not in sys.path:
    sys.path.append(dir_parent)

# set region and model name
aws_region = "us-west-2"
model_name = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# Set bedrock configs
bedrock_config = Config(
    connect_timeout=3000, read_timeout=3000, retries={"max_attempts": 10, "mode": "standard"}
)

# Functions
def add_history(role, content):

    message = bedrock_utils.get_message_from_string(
        role=role,
        string=content
    )
    st.session_state["messages"].append(message)

T = TypeVar("T")
def get_streamlit_cb(parent_container: DeltaGenerator):
    
    def decor(fn: Callable[..., T]) -> Callable[..., T]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> T:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = StreamlitCallbackHandler(parent_container=parent_container)

    for name, fn in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if name.startswith("on_"):
            setattr(st_cb, name, decor(fn))

    return st_cb

def display_chat_history():

    node_names = ["Agent"]
    st.session_state["history_ask"].append(st.session_state["recent_ask"])

    recent_answer = {}
    for node_name in node_names:
        st.session_state["history_answer"].append(recent_answer)

    for user, assistant in zip(st.session_state["history_ask"], st.session_state["history_answer"]):
        with st.chat_message("user"):
            st.write(user)

        tabs = st.tabs(node_names)
        with st.chat_message("assistant"):
            for tab, node_name in zip(tabs, node_names):
                with tab:
                    st.write(assistant[node_name])

####################### Initialization ###############################
# Store the initial value of widgets in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "analyzer" not in st.session_state:
    st.session_state["analyzer"] = DrugDiscoveryAssistantAgent()

####################### Application ###############################
st.set_page_config(page_title="PharmAgent AI - Your Intelligent Drug Discovery Assistant 💬", page_icon="💬", layout="centered") ## layout [centered or wide]
st.title("PharmAgent AI - Your Intelligent Drug Discovery Assistant 💬")
st.markdown('''- This chatbot is implemented using Amazon Bedrock Claude v3.5 Sonnet.''')

if len(st.session_state["messages"]) > 0: display_chat_history()

if user_input := st.chat_input():
    
    st.chat_message("user").write(user_input)
    st.session_state["recent_ask"] = user_input
    
    tab1 = st.tabs(["agent"])
    tabs = {
        "agent": tab1,
    }
    
    with st.chat_message("assistant"):
        with st.spinner(f'Thinking...'):
            st_callback = get_streamlit_cb(st.container())
            st.session_state["tabs"] = tabs
            st.session_state["analyzer"].chat(
                message=user_input,
                # st_callback=st_callback
            )
            st.write("Done")