# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:09:12 2022

@author: dship
"""

import os
import shutil
import glob
import configparser
import cv2
import streamlit as st
import torch
import pandas as pd
import numpy as np

# PyPI version contains bug, use fix below. pip install image-quality then use
# fix below.
#   import imquality.brisque as brisque
# https://learnopencv.com/image-quality-assessment-brisque/
import brisque

import warnings

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
from enum import Enum
from stqdm import stqdm
from PIL import Image as Img
from sklearn.cluster import DBSCAN
from dface import MTCNN, FaceNet
from utils import crop_face

#---------- Does not work as expected
import base64
from io import BytesIO

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_thumbnail(path):
    path = "\\\\?\\"+path # This "\\\\?\\" is used to prevent problems with long Windows paths
    i = Img.open(path)    
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
#----------

class Profile(Enum):
    LEFT_PROFILE = 0
    LEFT_ANGLE_PROFILE = 1
    FRONT_PROFILE = 2
    RIGHT_ANGLE_PROFILE = 3
    RIGHT_PROFILE = 4

class Cluster(object):
    def __init__(self):

        # supported image types for clustering
        self.supported_filetypes = [
            'jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff'
        ]

        # autodetect if gpu device driver (cuda) installed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        print(f"==> Forcing CPU use: self.device='{self.device}'")

        # initialize models
        #self.models_folder = os.path.abspath("./models")
        #mtcnn_model = self.models_folder + '/mtcnn.pt'
        #facenet_model = self.models_folder + '/facenet.pt'
        #self.mtcnn = MTCNN(self.device, model=mtcnn_model)
        #self.facenet = FaceNet(self.device, model=facenet_model)
        
        # initialize models
        # use for low-side streamlit cloud, otherwise comment out - this will auto-download models
        self.mtcnn = MTCNN(self.device)
        self.facenet = FaceNet(self.device)

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

    def __get_images(self, image_path):
        self.images = glob.glob(image_path + '/*')
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
            # image_names.append(os.path.basename(os.path.splitext(image)[0]))
            image_names.append(image)
            vid.release()

        return frames, image_names

    def __find_pose(self, points):
        X=points[0:5]
        Y=points[5:10]
    
        angle=np.arctan((Y[1]-Y[0])/(X[1]-X[0]))/np.pi*180
        alpha=np.cos(np.deg2rad(angle))
        beta=np.sin(np.deg2rad(angle))
        
        # compensate for roll: rotate points (landmarks) so that both the eyes are
        # alligned horizontally 
        Xr=np.zeros((5))
        Yr=np.zeros((5))
        for i in range(5):
            Xr[i]=alpha*X[i]+beta*Y[i]+(1-alpha)*X[2]-beta*Y[2]
            Yr[i]=-beta*X[i]+alpha*Y[i]+beta*X[2]+(1-alpha)*Y[2]
    
        # average distance between eyes and mouth
        dXtot=(Xr[1]-Xr[0]+Xr[4]-Xr[3])/2
        dYtot=(Yr[3]-Yr[0]+Yr[4]-Yr[1])/2
    
        # average distance between nose and eyes
        dXnose=(Xr[1]-Xr[2]+Xr[4]-Xr[2])/2
        dYnose=(Yr[3]-Yr[2]+Yr[4]-Yr[2])/2
    
        # relative rotation 0% is frontal 100% is profile
        # Xfrontal=np.abs(np.clip(-90+90/0.5*dXnose/dXtot,-90,90))
        # Yfrontal=np.abs(np.clip(-90+90/0.5*dYnose/dYtot,-90,90))
        Xfrontal=np.clip(-90+90/0.5*dXnose/dXtot,-90,90)
        Yfrontal=np.clip(-90+90/0.5*dYnose/dYtot,-90,90)

        roll = angle
        yaw = Xfrontal
        pitch = Yfrontal
    
        return yaw, pitch, roll

    def plot_similarity_grid(self, cos_similarity, input_size):
        n = len(cos_similarity)
        rows = []
        for i in range(n):
            row = []
            for j in range(n):
                # create small colorful image from value in distance matrix
                value = cos_similarity[i][j]
                cell = np.empty(input_size)
                cell.fill(value)
                cell = (cell * 255).astype(np.uint8)

                # color depends on value: blue is closer to 0, green is closer to 1
                img = cv2.applyColorMap(cell, cv2.COLORMAP_WINTER)
    
                # add distance value as text centered on image
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"{value:.2f}"
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = (img.shape[1] - textsize[0]) // 2
                text_y = (img.shape[0] + textsize[1]) // 2
                cv2.putText(
                    img, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA,
                )
                row.append(img)
            rows.append(np.concatenate(row, axis=1))
        grid = np.concatenate(rows)
        return grid
        
    def visualize_similarity(self, input_size=[128, 128]):
        """
        Plot a similarity matrix of input images

        Bounding boxes is a list of four coordinates (x1, y1, x2, y2)
        for each face detected in the frame. Probabilities is a list of
        confidence scores between 0 and 1 for each face. Landmarks is a
        list of facial landmark points. There are five points for each
        face.
        """
        # dface mtcnn needs to treat each image as a video frame
        # read images from directoruy using cv2 video capture        
        self.frames, self.names = self.__get_images(self.imgpath)

        # calculate image hash values
        # for frame in self.frames:
        #     frame_hash = cv2.img_hash.averageHash(frame)[0]
        #     frame_hash = ''.join(hex(i)[2:] for i in frame_hash)
        #     st.write(frame_hash)
        
        # read frames needs to be a video from for the dface mtcnn algorithm to work
        #   coordinates for bounding box and landmarks, and confidences score
        self.detected = self.mtcnn.detect(self.frames)
        #st.write(self.detected)

        # lists that will contain actual face images and name reference
        faces = []  # bounding box cropped face images

        self.detected_images = []
        self.detected_names = []
        self.frame_indices = []
        
        # TODO: add stqdm progress bar
        for idx, detected in enumerate(self.detected):
            #if idx == 0:
            #   st.write(self.names[idx], detected)
            #   st.write(self.frames[idx].shape)
               
            if detected is None:
                #TODO: remove undetected image
                continue
            x = detected[0][0][0]
            y = detected[0][0][1]
            w = detected[0][0][2]
            h = detected[0][0][3]
            box = x, y, w, h
            #st.write(names[i], box)

            face = crop_face(self.frames[idx], box)
            #face = frames[i][math.floor(y):math.ceil(h), math.floor(x):math.ceil(w)]

            faces.append(face)
            
            self.detected_images.append(self.images[idx])
            self.detected_names.append(self.names[idx])            
            self.frame_indices.append(idx)
        
        # calculate face embedddings using dface facenet
        self.embeddings = self.facenet.embedding(faces)
        #st.write(len(self.embeddings))

        # calculate cosine similarity matrix
        self.cos_similarity = np.dot(self.embeddings, self.embeddings.T)
        self.cos_similarity = self.cos_similarity.clip(min=0, max=1)
        #st.write(self.cos_similarity)
        #st.write(self.frame_indices)
        
        # plot colorful grid from pair distance values in similarity matrix
        similarity_grid = self.plot_similarity_grid(self.cos_similarity, input_size)
        
        # pad similarity grid with images of faces
        horizontal_grid = np.hstack([self.frames[i] for i in self.frame_indices])
        vertical_grid = np.vstack([self.frames[i] for i in self.frame_indices])
        zeros = np.zeros((*input_size, 3))
        vertical_grid = np.vstack((zeros, vertical_grid))
        result = np.vstack((horizontal_grid, similarity_grid))
        result = np.hstack((vertical_grid, result))
    
        # create similarity directory
        self.similarity_path = self.imgpath + "/similarity/"
        if not os.path.isdir(self.similarity_path):
            os.mkdir(self.similarity_path)
    
        # Save image file
        cv2.imwrite(self.similarity_path + "/test.jpg", result)
        
        # plot image to web page
        st.subheader('Similarity Matrix')
        st.markdown(
            """
            Calculate the cosine distance between facial embeddings to determine
            similarilary between two or more persons. Best results are those of
            a frontal profile or a slightly turned left or right profile. Extreme
            profiles or occlussions will result in low similarity scores even
            for the same person.
            """)
        image = Img.open(self.similarity_path + "/test.jpg")
        st.image(image)
        
    def find_pose(self):
        """
        """
        #st.write(len(self.detected))
        
        for idx, detected in enumerate(self.detected):
            if detected is None:
                continue

            landmarks = [
                # x
                detected[2][0][0][0], # left eye
                detected[2][0][1][0], # right eye
                detected[2][0][2][0], # nose
                detected[2][0][3][0], # mouth left
                detected[2][0][4][0], # mouth right
                # y
                detected[2][0][0][1], # left eye
                detected[2][0][1][1], # right eye
                detected[2][0][2][1], # nose
                detected[2][0][3][1], # mouth left
                detected[2][0][4][1], # mouth right
            ]
                        
            yaw, pitch, roll = self.__find_pose(landmarks)
            
            metadata={
#                'Image': self.images[idx],
                'Image': self.frames[idx],
#                'Image': "https://en.wikipedia.org/wiki/File:American_Eskimo_Dog.jpg", #self.images[idx], #self.frames[idx],
                'Name': self.images[idx], #self.names[idx],
                'Height': self.frames[idx].shape[0],
                'Width': self.frames[idx].shape[1],
                'Quality': brisque.score(Img.open(self.images[idx])),
                'Yaw': yaw,
                'Pitch': pitch,
                'Roll': roll,
                'Confidence': detected[1][0],
                'IPD': detected[2][0][1][0] - detected[2][0][0][0],
                'BoxXY': tuple(detected[0][0]),
                'Left Eye': tuple(detected[2][0][0]),
                'Right Eye': tuple(detected[2][0][0]),
                'Nose': tuple(detected[2][0][0]),
                'Mouth Left': tuple(detected[2][0][0]),
                'Mouth Right': tuple(detected[2][0][0])
            }
        
            self.image_df = self.image_df.append(metadata, ignore_index=True)

        st.subheader("Image Metrics")
        gb = GridOptionsBuilder.from_dataframe(self.image_df)
        thumbnail_image = JsCode("""function (params) {
                var element = document.createElement("span");
                var imageElement = document.createElement("img");
            
                if (params.data.Image) {
                    imageElement.src = params.data.Image;
                    imageElement.width="20";
                } else {
                    imageElement.src = "";
                }
                element.appendChild(imageElement);
                element.appendChild(document.createTextNode(params.value));
                return element;
                }""")
        gb.configure_column('Name', cellRenderer=thumbnail_image)
        gridOptions = gb.build()
        AgGrid(self.image_df,
               gridOptions=gridOptions,
               fit_columns_on_grid_load=True,
               allow_unsafe_jscode=True,
               enable_enterprise_modules=True)
            # TODO: make dataframe
            # image, name, profile category, yaw, pitch, roll, confidence, bounding box, left eye, right eye, nose, mouth left, mouth right

    def DBSCAN_sort(self):
        """
        Function to form clusters using the DBSCAN clustering algorithm
        """
        # # dbscan = DBSCAN(eps=self.maximum_distance, min_samples=self.minimum_samples, metric=self.metric)
        #dbscan = DBSCAN(eps=0.32, min_samples=2, metric='cosine', n_jobs=-1)
        #labels = dbscan.fit_predict(self.embeddings)
        #st.write(labels)
        dbscan = DBSCAN(eps=self.maximum_distance, min_samples=self.minimum_samples, metric=self.metric, n_jobs=-1).fit(self.embeddings)
        labels = dbscan.labels_
        #st.write(labels)
        
        # create a sorted directory
        self.cluster_path = self.imgpath + "/clustered/"
        if not os.path.isdir(self.cluster_path):
            os.mkdir(self.cluster_path)

        #st.write('max=', max(labels))
        #st.write('min=', min(labels))
        
        # create cluster subfolders
        for cluster_id in range(min(labels), max(labels) + 1):
            os.mkdir(self.cluster_path + str(cluster_id)) 

        for i in stqdm(range(len(labels)),
                       #st_container=st.sidebar,
                       leave=True,
                       desc='Clustering Results: ',
                       gui=True):

            shutil.copy(self.detected_images[i], self.cluster_path + str(labels[i]))

    def cluster_sort(self):
        """
        Function to form clusters based on the Cosine Similarity Matrix
        """
        #st.write(self.cos_similarity)
        #st.write(len(self.frame_indices), len(self.detected_images))
        #st.write(self.frame_indices)

        # set cluster threshold
        self.threshold = 0.52

        # create a sorted directory
        self.cluster_path = self.imgpath + "/clustered/"
        if not os.path.isdir(self.cluster_path):
            os.mkdir(self.cluster_path)

        # cluster counter
        cluster_index = 0
        
        # images are ordered in self.frames_indices
        #for row in range(len(self.frame_indices)):
        for row in stqdm(range(len(self.frame_indices)),
                        #st_container=st.sidebar,
                        leave=True,
                        desc='Clustering Results: ',
                        gui=True):

            # create clustered images folder
            self.cluster_image_path = self.cluster_path + str(cluster_index)
            if not os.path.isdir(self.cluster_image_path):
                os.mkdir(self.cluster_image_path)
                
            # copy images to clustered image folder that meets or exceeds threshold
            for col in range(len(self.frame_indices)):
                #st.write(self.cos_similarity[row][col])
                if self.cos_similarity[row][col] >= self.threshold:
                    if self.frame_indices[col] != -1:
                        shutil.copy(self.detected_images[col], self.cluster_image_path)
                        self.frame_indices[col] = -1
                    
            # create new cluster if the previous cluster has data
            listdir = os.listdir(self.cluster_image_path)
            if len(listdir) != 0:

                cluster_index = cluster_index + 1           

        # check and remove the last created cluster folder (should not have anything in it)
        self.cluster_image_path = self.cluster_path + str(cluster_index)
        listdir = os.listdir(self.cluster_image_path)
        #st.write(self.cluster_image_path, listdir)
        if len(listdir) == 0:
            shutil.rmtree(self.cluster_image_path)

        # create negative cluster folder
        self.cluster_image_path = self.cluster_path + str(-1)
        if not os.path.isdir(self.cluster_image_path):
            os.mkdir(self.cluster_image_path)

        # move undetected images to the negative cluster folder
        for idx in range(len(self.detected)):
            if self.detected[idx] is None:
                shutil.copy(self.images[idx], self.cluster_image_path)
                #st.write('NULL FOUND!', idx)                

    def pose_sort(self):
        """
        Grab images in each cluster folder and sort by pose.
        """
        # get each cluster put into list
        try:
            rootdir = self.cluster_path
            clusters = [os.path.abspath(d.path) for d in os.scandir(rootdir) if d.is_dir()]
        except:
            clusters = [self.imgpath] # the master directory

        #st.write(clusters)

        # sort images in each cluster by pose
        for cluster in clusters:

            # create subfolders in each cluster folder for pose sorting
            folder_left = cluster + "/left/"
            if not os.path.isdir(folder_left):
                os.mkdir(folder_left)

            folder_left_angle = cluster + "/left-angle/"
            if not os.path.isdir(folder_left_angle):
                os.mkdir(folder_left_angle)
            
            folder_front = cluster + "/front/"
            if not os.path.isdir(folder_front):
                os.mkdir(folder_front)

            folder_right_angle = cluster + "/right-angle/"
            if not os.path.isdir(folder_right_angle):
                os.mkdir(folder_right_angle)

            folder_right = cluster + "/right/"
            if not os.path.isdir(folder_right):
                os.mkdir(folder_right)

            # interpret images as a video frame for use with the dface mtcnn algorithm
            frames, names = self.__get_images(cluster)
            
            # detect faces in the video frame
            mtcnn_detected = self.mtcnn.detect(frames)
            
            # calculate pose for each detected face using facial landmarks            
            for idx, detected in enumerate(mtcnn_detected):
                if detected is None:
                    continue
    
                landmarks = [
                    # x
                    detected[2][0][0][0], # left eye
                    detected[2][0][1][0], # right eye
                    detected[2][0][2][0], # nose
                    detected[2][0][3][0], # mouth left
                    detected[2][0][4][0], # mouth right
                    # y
                    detected[2][0][0][1], # left eye
                    detected[2][0][1][1], # right eye
                    detected[2][0][2][1], # nose
                    detected[2][0][3][1], # mouth left
                    detected[2][0][4][1], # mouth right
                ]
                            
                yaw, pitch, roll = self.__find_pose(landmarks)
                
                # metadata={
                #     'Cluster': os.path.basename(cluster),
                #     'Name': os.path.basename(names[idx]),
                #     'Yaw': yaw,
                #     'Pitch': pitch,
                #     'Roll': roll,
                # }                
                # st.write(metadata)
        
                # copy image to the appropriate profile subfolder1
                if -100 <= yaw < -85:
                    # shutil.copy(names[idx], folder_right)
                    shutil.move(names[idx], folder_right)
                elif -85 <= yaw < -15:
                    # shutil.copy(names[idx], folder_right_angle)
                    shutil.move(names[idx], folder_right_angle)
                elif -15 <= yaw < 15:
                    # shutil.copy(names[idx], folder_front)
                    shutil.move(names[idx], folder_front)
                elif 15 <= yaw < 85:
                    # shutil.copy(names[idx], folder_left_angle)
                    shutil.move(names[idx], folder_left_angle)
                elif 85 <= yaw <= 100:
                    # shutil.copy(names[idx], folder_left)
                    shutil.move(names[idx], folder_left)

    def copy_face_images(self):
        """
        Copy uploaded images to a subfolder.
        """
        self.imgpath = os.path.abspath(self.output_folder) + '/master/'
        
        # TODO: Do we want to destroy the cluster folder or just add to it?
        if os.path.exists(self.imgpath):
            shutil.rmtree(self.imgpath)
            os.makedirs(self.imgpath)
        else:
            os.makedirs(self.imgpath)

        for uploaded_file in self.uploaded_files:
            with open(self.imgpath + uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())
                
        del self.uploaded_files

    def run(self):
        """
        """        
        # set streamlit page defaults
        st.set_page_config(
            layout = 'wide', # centered, wide, dashboard
            initial_sidebar_state = 'auto', # auto, expanded, collapsed
            page_title = 'BATMAN+',
            page_icon = Img.open("assets/baticon.png") #':eyes:' # https://emojipedia.org/shortcodes/
        )

        # set title and format
        st.markdown(""" <style> .font {font-size:60px ; font-family: 'Sans-serif'; color: blue;} </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Biometric Analysis Tool for Media ANalytics</p>', unsafe_allow_html=True)

        # sidebar widgets
        st.sidebar.subheader('Cluster and Pose Settings')
        self.output_folder = st.sidebar.text_input('Output Directory:', value=os.path.abspath("output"), help="Output directory where cluster and pose results are stored.")
        # self.confidence = st.sidebar.slider('Confidence:', min_value=0.0, max_value=1.0, step=0.01, value=0.90, help="Face detection confidence value.")
        self.maximum_distance = st.sidebar.slider('EPS:', min_value=0.0, max_value=1.0, step=0.01, value=0.32, help="Maximum distance between two samples to be considered in the same cluster.")
        self.minimum_samples = st.sidebar.slider('Min Samples:', min_value=0, max_value=10, step=1, value=2, help="Minimum number of samples in a cluster for point to be considered as a core point.")

        # TODO: Make this a selectbox
        self.metric = st.sidebar.selectbox('Distance Metric:', ('cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'), index=1, help="Metric to use when calculating distance between instances in a feature array.")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            self.pose = st.checkbox("Sort by Pose", value=False, help="Sort images by pose.")
        with col2:
            self.cluster = st.checkbox("Cluster", value=True, help="Cluster images using DBSCAN algorithm.")

        # media input
        st.subheader('Image Files')
        self.uploaded_files = st.file_uploader('Select face image files.', type=self.supported_filetypes, accept_multiple_files=True)

        if self.uploaded_files != []:
            self.copy_face_images()

            self.visualize_similarity()
            self.find_pose()

            if self.cluster:
                self.DBSCAN_sort()
                #self.cluster_sort()

            if self.pose:
                self.pose_sort()

if __name__ == '__main__':
    c = Cluster()
    c.run()