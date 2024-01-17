

import streamlit as st
import os
import requests
from urllib import request
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from utils.io_tools import download_to_archive, clear_info
from utils import session
from utils.logging import timestamp
import subprocess

if 'query_params' not in st.session_state:
        st.session_state.query_params = {}
        st.session_state.query_params['console'] = False
        st.session_state.query_params['resources'] = False
        st.session_state.query_params['cache'] = False

if 'show_resource_usage' not in st.session_state:
    st.session_state.show_resource_usage = False
else:
    st.session_state.show_resource_usage = False

if 'show_console' not in st.session_state:
    st.session_state.show_console = False

if 'console_out' not in st.session_state:
    st.session_state.console_out = ''

if 'console_in' not in st.session_state:
    st.session_state.console_in = ''

if 'cache_checked' not in st.session_state:
    st.session_state.cache_checked = False

if 'data_checked' not in st.session_state:
    st.session_state.data_checked = False

if 'low_resources' not in st.session_state:
    st.session_state.low_resources = False
    
if 'wget_failed' not in st.session_state:
    st.session_state.wget_failed = False

if "target_url" not in st.session_state:
    st.session_state.target_url = ""

if "zip_filename" not in st.session_state:
    st.session_state.zip_filename = ""

if "urls" not in st.session_state:
    st.session_state.urls = []    

if "selected_targets_info" not in st.session_state:
    st.session_state.selected_targets_info = [] 

if "filenames" not in st.session_state:
    st.session_state.filenames = []  

if "target_info" not in st.session_state:
    st.session_state.target_info = None

if "message" not in st.session_state:
    st.session_state.message = ""

if "count_downloaded" not in st.session_state:
    st.session_state.count_downloaded = 0

if 'called_process_error' not in st.session_state:
    st.session_state.called_process_error = None

def run_command():
    print(f'[{timestamp()}] st.session_state.console_in: {st.session_state.console_in}')
    try:
        st.session_state.console_out = str(subprocess.check_output(st.session_state.console_in, shell=True, text=True))
        st.session_state.console_out_timestamp = f'{timestamp()}'
    except subprocess.CalledProcessError as e:
        st.session_state.console_out = f'exited with error\nreturncode: {e.returncode}\ncmd: {e.cmd}\noutput: {e.output}\nstderr: {e.stderr}'
        st.session_state.console_out_timestamp = f'{timestamp()}'

    print(f'[{timestamp()}] st.session_state.console_out: {st.session_state.console_out}')


        placeholder = st.empty()
        if st.session_state.show_console:
            with placeholder.container():
                with st.expander("console", expanded=True):
                    with st.form('console'):
                        command = st.text_input(f'[{pid}] {timestamp()}', str(st.session_state.console_in), key="console_in")
                        submitted = st.form_submit_button('run', help="coming soon", on_click=run_command)

                        st.write(f'IN: {command}')
                        st.text(f'OUT:\n{st.session_state.console_out}')
                    file_name = st.text_input("File Name", "")
                    if os.path.isfile(file_name):
                        button = st.download_button(label="Download File", data=Path(file_name).read_bytes(), file_name=file_name, key="console_download")
        else:
             placeholder.empty()

class URL_Query_Extras():
        self.query_params = {}
        self.query_params['console'] = False
        self.query_params['resources'] = False
        self.query_params['cache'] = False

        self.show_resource_usage = False


        self.cache_checked = False

        self.data_checked = False
        self.low_resources = False
        self.wget_failed = False

        self.target_url = ""
        self.called_process_error = None

class Console():

    def __init__(self):


        self.show_console = False

        self.console_out = ''

        self.console_in = ''
        self.console_out_timestamp = None


    def run_command(self, command, shell=True, text=True):
        self.console_in = command
        print(f'[{timestamp()}] self.console_in: {self.console_in}')
        try:
            self.console_out = str(subprocess.check_output(self.console_in, shell=shell, text=text))

            self.console_out_timestamp = f'{timestamp()}'
        except subprocess.CalledProcessError as e:
            self.console_out = f'exited with error\nreturncode: {e.returncode}\ncmd: {e.cmd}\noutput: {e.output}\nstderr: {e.stderr}'
            self.console_out_timestamp = f'{timestamp()}'

        print(f'[{timestamp()}] self.console_out: {self.console_out}')


