import os
import cv2
import streamlit as st
import warnings

from PIL import Image as Img
from timeline.generator import generate_timeline
from streamlit_terran_timeline import terran_timeline

# page configuration
st.set_page_config(
    layout = 'wide', # centered, wide, dashboard
    initial_sidebar_state = 'auto', # auto, expanded, collapsed
    page_title = 'Media Extractor',
    page_icon = Img.open(r"./assets/baticon.png") #':eyes:' # https://emojipedia.org/shortcodes/
)

#sidebar settings
st.sidebar.subheader('Video Timeline Settings')
output_folder = st.sidebar.text_input('Timeline Directory:', value=os.path.abspath(r"output"), key='media_output', help="Directory location containing the timeline subfolder.")
reference_folder = st.sidebar.text_input('Reference Directory:', value="", key='reference_output', help="A path to a folder containing images of faces to look for in the video. If the value is empty, then it'll automatically collect the faces as we read the video and generate their timeline automatically.")
batch_size = st.sidebar.slider('Batch Size:', min_value=1, max_value=64, step=1, value=32, key='batch_size', help="How many frames to process at once.")
duration = st.sidebar.slider('Duration:', min_value=0, max_value=300, step=1, value=0, key='duration', help="How many seconds of the video should be processed. If equals to 0 then all the video is processed.")
# TODO: get max video time
start_time = st.sidebar.slider('Start Time:', min_value=0, max_value=300, step=1, value=0, key='start_time', help="The starting time (in seconds) to begin the timeline generation.")
framerate = st.sidebar.slider('Framerate:', min_value=1, max_value=30, step=1, value=8, key='framerate', help="How many frames per second we should process.")
thumbnail_rate = st.sidebar.slider('Thumbnail Rate:', min_value=0, max_value=30, step=1, value=1, key='thumbnail_rate', help="Collect a thumbnail of the video for every X seconds. If 0, it won't collect thumbnails.")
appearance_threshold = st.sidebar.slider('Appearance Threshold:', min_value=1, max_value=10, step=1, value=5, key='appearance_threshold', help="If a face appears more then this amount it will be considered for the timeline.")
similarity_threshold = st.sidebar.slider('Similarity Threshold:', min_value=0.0, max_value=1.0, step=0.01, value=0.75, key='similarity_threshold', help="A distance value for when two faces are the same.")

if duration == 0:
    duration = None
if thumbnail_rate == 0:
    thumbnail_rate = None
    
st.sidebar.markdown("---")

# set title and format
st.markdown(""" <style> .font {font-size:60px ; font-family: 'Sans-serif'; color: blue;} </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Biometric Analysis Tool for Media ANalytics</p>', unsafe_allow_html=True)
st.write(
    "Video timeline chart of faces detected on videos. Videos can be selected from **multiple sources** such as YouTube and almost any video streaming platform, or any local file. Video timeline is based on the Python library called Terran, a human perception library, that can perform face-detection (retinaface), face-recognition (arcface), and pose estimation (openpose)."
)

st.subheader('Media Input')
supported_filetypes = ['mp4', 'avi', 'webm', 'wmv', 'jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff']
video = st.file_uploader("Choose a media file (image, video, or document)", type=supported_filetypes, accept_multiple_files=False)
#st.write(video.name)

# generate faces timeline chart
if video is not None:
    # create output folder
    output_folder = os.path.abspath(output_folder)
    #st.write(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # create timeline folder
    timeline_folder = output_folder + '/timeline'
    #st.write(timeline_folder)
    if not os.path.exists(timeline_folder):
        os.makedirs(timeline_folder)

    # copy buffer to output folder
    video_path = output_folder + '/' + video.name        
    #st.write(video_path)
    with open(video_path, "wb") as f:
        f.write(video.read())

    st.subheader("Faces Timeline Chart")
             
    @st.cache(persist=True, ttl=86_400, suppress_st_warning=True, show_spinner=False)
    def _generate_timeline(video_path):
        timeline = generate_timeline(
            video_src=video_path,
            batch_size=batch_size,
            duration=duration,
            start_time=start_time,
            framerate=framerate,
            thumbnail_rate=thumbnail_rate,
            output_directory=timeline_folder,
            ref_directory=reference_folder,
            appearence_threshold=appearance_threshold,
            similarity_threshold=similarity_threshold,
        )

        return timeline


    with st.spinner("Generating timeline"):
        timeline = _generate_timeline(video_path)
    #st.write(timeline)
    
    start_time = terran_timeline(timeline)

    st.write(f'Video playback start time: {int(start_time)}')

    with open(video_path, 'rb') as v:
        st.video(v, start_time=int(start_time))
