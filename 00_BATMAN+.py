# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:34:55 2022

@author: dship
"""
import os
import streamlit as st
from PIL import Image

class Batman(object):
    """
    Biometric Analytic Tool for Media Extraction.
    Web frontend using streamlit.
    """
    def __init__(self):
        """
        Initialize streamlit configuration options.
        """
        # set streamlit page defaults
        # st.set_page_config(
        #     layout = 'wide', # centered, wide, dashboard
        #     initial_sidebar_state = 'auto', # auto, expanded, collapsed
        #     page_title = 'BATMAN+',
        #     page_icon = Image.open(os.path.abspath(r".\assets\baticon.png")) #':eyes:' # https://emojipedia.org/shortcodes/
        # )

        # #st.sidebar.image("./assets/baticon.png")
        # st.sidebar.success("Select application above.")
        # #st.sidebar.markdown("---")

        # disables hamburger menu on top right of page and footer
        #st.markdown(""" <style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style> """, unsafe_allow_html=True)
        
    def run(self):
        """
        Starts the streamlit frontend with a sidebar, title, and custom page
        links.
        """
        #print(os.getcwd())
        #print(os.path.abspath('assets/baticon.png'))
        st.set_page_config(
            layout = 'wide', # centered, wide, dashboard
            initial_sidebar_state = 'auto', # auto, expanded, collapsed
            page_title = 'BATMAN+',
            page_icon = Image.open(r"./assets/baticon.png") #':eyes:' # https://emojipedia.org/shortcodes/
            #page_icon = Image.open('baticon.png') #':eyes:' # https://emojipedia.org/shortcodes/
            #page_icon = Image.open(r"C:\Users\dship\Desktop\bio_final\batman+\assets\baticon.png") #':eyes:' # https://emojipedia.org/shortcodes/
            #Apage_icon = ':bat:' # https://emojipedia.org/shortcodes/
        )

        # set title and format
        st.markdown(""" <style> .font {font-size:60px ; font-family: 'Sans-serif'; color: blue;} </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Biometric Analysis Tool for Media ANalytics</p>', unsafe_allow_html=True)

        st.image(r"./assets/baticon.png")
        st.write("## Welcome to BATMAN+!")
        st.markdown(
            """
            BATMAN is an forensic analysis tool used to help analysts process image data contained in files.
            
            ### Automatic extraction pipeline:
               - Extract images from documents and files
               - Crop faces from extracted images
               - Cluster faces by identity
               - Sort faces by pose 
            """)

if __name__ == '__main__':
    bm = Batman()
    bm.run()
