# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:09:12 2022

@author: BRL
"""

import os
import shutil
import glob
import cv2
import streamlit as st
#import torch
import pandas as pd
#import numpy as np

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
#from enum import Enum
from stqdm import stqdm
from PIL import Image as Img
#from dface import MTCNN, FaceNet
#from utils import crop_face
from fer import FER
from feat.detector import Detector

class FaceExpressionAnalysis(object):
    def __init__(self):
        """
        """
        # supported image types for clustering
        self.supported_filetypes = [
            'jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff'
        ]

        # yaw, pitch, roll table
        self.image_df = pd.DataFrame(columns=['Image',
                                              'Name',
                                              'Height',
                                              'Width',
                                              'Quality',
                                              'Yaw',
                                              'Pitch',
                                              'Roll',
                                              'Confidence',
                                              'IPD',
                                              'BoxXY',
                                              'Left Eye',
                                              'Right Eye',
                                              'Nose',
                                              'Mouth Left',
                                              'Mouth Right'])

        self.emotions_df = pd.DataFrame(columns=['Image',
                                                 'Name',
                                                 'Box',
                                                 'TopEmotion',
                                                 'Angry',
                                                 'Disgust',
                                                 'Fear',
                                                 'Happy',
                                                 'Sad',
                                                 'Surprise',
                                                 'Neutral'])

        self.fau_df = pd.DataFrame()

    def __get_images(self):
        self.images = glob.glob(self.imgpath + '/*')
        image_names = []
        frames = []
        for idx, image in enumerate(self.images):
            vid = cv2.VideoCapture(image)
            ok = vid.grab()
            ok, frm = vid.retrieve()
            if not ok:
                continue
            # frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            frm = cv2.resize(frm, (128, 128)) # 96, 96
            frames.append(frm)
            image_names.append(os.path.basename(os.path.splitext(image)[0]))
            vid.release()

        return frames, image_names

    def copy_face_images(self):
        """
        Copy uploaded images to a subfolder.
        """
        self.imgpath = os.path.abspath(self.output_folder) + '/emotion/'
        
        if os.path.exists(self.imgpath):
            shutil.rmtree(self.imgpath)
            os.makedirs(self.imgpath)
        else:
            os.makedirs(self.imgpath)

        for uploaded_file in self.uploaded_files:
            with open(self.imgpath + uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())
                
        del self.uploaded_files
        
    def get_face_emotion(self):
        
        # faces by default are detected using OpenCV's Haar Cascade classifier
        #   To use the more accurate MTCNN network add parameter mtcnn=True
        detector = FER(mtcnn=True)

        self.frames, self.names = self.__get_images()
        
        for idx, frame in enumerate(self.frames):
            #with open(cv2.imread(frame)) as f:
            fer_out = detector.detect_emotions(frame)
            #st.write(self.names[idx], fer_out)

            if fer_out != []:
                metadata={
                    'Image': self.frames[idx],
                    'Name': self.names[idx],
                    'Box': fer_out[0]['box'],
                    'TopEmotion': detector.top_emotion(frame),
                    'Angry': fer_out[0]['emotions']['angry'],
                    'Disgust': fer_out[0]['emotions']['disgust'],
                    'Fear': fer_out[0]['emotions']['fear'],
                    'Happy': fer_out[0]['emotions']['happy'],
                    'Sad': fer_out[0]['emotions']['sad'],
                    'Surprise': fer_out[0]['emotions']['surprise'],
                    'Neutral': fer_out[0]['emotions']['neutral']
                }
            else:
                metadata={
                    'Image': self.frames[idx],
                    'Name': self.names[idx],
                    'Box': '',
                    'TopEmotion': 'NOT DETECTED',
                    'Angry': '',
                    'Disgust': '',
                    'Fear': '',
                    'Happy': '',
                    'Sad': '',
                    'Surprise': '',
                    'Neutral': ''
                }

            self.emotions_df = self.emotions_df.append(metadata, ignore_index=True)

        st.subheader('Facial Expression Recognition')
        st.markdown(
            """
            Facial expression analysis using FER (Face Expression Recognition)
            Python library. https://github.com/justinshenk/fer
            """)
        AgGrid(self.emotions_df,
               #gridOptions=gridOptions,
               fit_columns_on_grid_load=True,
               allow_unsafe_jscode=True,
               enable_enterprise_modules=True)
        
    def facial_action_units(self):
        """
        """
        detector = Detector(au_model='svm',
                            emotion_model='resmasknet',
                            face_model='mtcnn',
                            facepose_model='img2pose',
                            landmark_model='mobilefacenet')

        #detector = Detector(au_model='svm',
        #                    emotion_model='fer',
        #                    face_model='mtcnn',
        #                    facepose_model='img2pose',
        #                    landmark_model='mobilefacenet')
        
        self.frames, self.names = self.__get_images()
        
        #detector.change_model(face_model="MTCNN")
        #detector.change_model()

        st.subheader('Facial Expression Analysis')
        st.markdown(
            """
            Facial expression analysis using Py-Feat Facial Expression Analysis
            Toolbox. Py-Feat provides a comprehensive API to easily detect facial
            expresions.
            
            Reference: https://py-feat.org/pages/intro.html
            """)

        # for idx, frame in enumerate(self.frames):
        # for image in self.images:
        #st.write(f'==> Number of images: {len(self.images)}')
        #st.write(f'==> {type(self.images)}')
        for it in stqdm(range(len(self.images)),
                       st_container=st.sidebar,
                       leave=True,
                       desc='Face Expression Analysis: ',
                       gui=True):
            
            #st.write(f'==> Image count: {it}')
            #st.write(self.images)

            output_filename = self.imgpath + os.path.basename(os.path.splitext(self.images[it])[0]) + '.csv'
            #st.write(output_filename)
            #st.write(self.images[it])
            fau_out = detector.detect_image(self.images[it], outputFname=output_filename)
            st.dataframe(fau_out)
            figs = fau_out.plot_detections(poses=True)
            figs[0].set_figheight(8)
            figs[0].set_figwidth(30)
            figs[0].savefig(self.imgpath + os.path.basename(os.path.splitext(self.images[it])[0]) + '.png')
            st.pyplot(figs[0])
            self.fau_df = self.fau_df.append(fau_out)
        
#        if self.fau_df.empty:
#            return

        # only display a subset of the 173 columns (the remaining are landmark points)
        aus_df = self.fau_df[['input',
                              'AU01',
                              'AU02',
                              'AU04',
                              'AU05',
                              'AU06',
                              'AU07',
                              'AU09',
                              'AU10',
                              'AU11',
                              'AU12',
                              'AU14',
                              'AU15',
                              'AU17',
                              'AU20',
                              'AU23',
                              'AU24',
                              'AU25',
                              'AU26',
                              'AU28',
                              'AU43']]

        facebox_df = self.fau_df[['input',
                                  'FaceRectX',
                                  'FaceRectY',
                                  'FaceRectWidth',
                                  'FaceRectHeight',
                                  'FaceScore']]

        emotions_df = self.fau_df[['input',
                                   'anger',
                                   'disgust',
                                   'fear',
                                   'happiness',
                                   'sadness',
                                   'surprise',
                                   'neutral']]

        facepose_df = self.fau_df[['input',
                                   'Pitch',
                                   'Roll',
                                   'Yaw']]
        
        st.markdown("""
            ##### Facial Action Units (FAU)
            
            Action Unit Reference: https://py-feat.org/pages/au_reference.html
        """)
        AgGrid(aus_df,
               #gridOptions=gridOptions,
               fit_columns_on_grid_load=False,
               allow_unsafe_jscode=True,
               enable_enterprise_modules=True)

        st.markdown('##### Facebox')
        AgGrid(facebox_df,
               #gridOptions=gridOptions,
               fit_columns_on_grid_load=True,
               allow_unsafe_jscode=True,
               enable_enterprise_modules=True)

        st.markdown('##### Emotions')
        AgGrid(emotions_df,
               #gridOptions=gridOptions,
               fit_columns_on_grid_load=True,
               allow_unsafe_jscode=True,
               enable_enterprise_modules=True)

        st.markdown('##### Face Pose')
        AgGrid(facepose_df,
               #gridOptions=gridOptions,
               fit_columns_on_grid_load=True,
               allow_unsafe_jscode=True,
               enable_enterprise_modules=True)

    def run(self):
        """
        """        
        # set streamlit page defaults
        st.set_page_config(
            layout = 'wide', # centered, wide, dashboard
            initial_sidebar_state = 'auto', # auto, expanded, collapsed
            page_title = 'BATMAN+',
            page_icon = Img.open("./assets/baticon.png") #':eyes:' # https://emojipedia.org/shortcodes/
        )

        # set title and format
        st.markdown(""" <style> .font {font-size:60px ; font-family: 'Sans-serif'; color: blue;} </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Biometric Analysis Tool for Media ANalytics</p>', unsafe_allow_html=True)

        # sidebar widgets
        st.sidebar.subheader('Face Expression Settings')
        self.output_folder = st.sidebar.text_input('Output Directory:', value="output", help="Output directory where cluster and pose results are stored.")

        # media input
        st.subheader('Image Files')
        self.uploaded_files = st.file_uploader('Select face image files.', type=self.supported_filetypes, accept_multiple_files=True)

        if self.uploaded_files != []:
            #st.write(self.uploaded_files[0])
            self.copy_face_images()
            #self.get_face_emotion()
            self.facial_action_units()

if __name__ == '__main__':
    f = FaceExpressionAnalysis()
    f.run()
