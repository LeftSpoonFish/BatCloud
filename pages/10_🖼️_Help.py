# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:09:12 2022

@author: dship
"""

import streamlit as st

class Help(object):
    def __init__(self):
        # set title and format
        st.markdown(""" <style> .font {font-size:60px ; font-family: 'Sans-serif'; color: blue;} </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Biometric Analysis Tool for Media ANalytics</p>', unsafe_allow_html=True)
        
        st.text('This is a help page. At some point, there will be help here. Right now, you are on your own!')
        
        # this is just to force the line separator below the application links in the sidebar
        st.sidebar.text('')

if __name__ == '__main__':
    h = Help()