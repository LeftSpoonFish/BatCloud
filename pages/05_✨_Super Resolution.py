# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:09:12 2022

@author: BRL
"""
import os
import io
import cv2
import base64
import numpy as np
import torch
import uuid
import streamlit as st
import streamlit.components.v1 as components
import urllib

from PIL import Image
from basicsr.utils import imwrite
from gfpgan import GFPGANer

def super_resolution(input_image, output_image, bg_upsampler='realesrgan'):
    """
    Inference demo for GFPGAN.
    """
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path="models/RealESRGAN_x2plus.pth",
                model=model,
                tile=400, # Tile size for background sampler, 0 for no tile during testing. Default: 400
                tile_pad=10,
                pre_pad=0,
                half=False)  # need to set False in CPU mode
    else:
        bg_upsampler = None
            
    # set up GFPGAN restorer
    restorer = GFPGANer(
        #model_path="models/GFPGANCleanv1-NoCE-C2.pth",
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)

    # restore faces and background if necessary
    iimg = cv2.imread(input_image, cv2.IMREAD_COLOR)
    cropped_faces, restored_faces, restored_img = restorer.enhance(iimg, has_aligned=True, only_center_face=True, paste_back=False)

    #st.write(type(cropped_faces)) # list
    #st.write(type(restored_faces)) # list
    #st.write(type(restored_img)) # list?

    imwrite(cropped_faces[0], output_image+"/cropped.jpg")
    imwrite(restored_faces[0], output_image+"/restored.jpg")
    #cv2.imwrite(output_image+"/restored.jpg", restored_img)
 
def pillow_to_base64(image: Image.Image):
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format="JPEG", subsampling=0, quality=100)
    img_bytes = in_mem_file.getvalue()  # bytes
    image_str = base64.b64encode(img_bytes).decode("utf-8")
    base64_src = f"data:image/jpg;base64,{image_str}"
    return base64_src

def local_file_to_base64(image_path: str):
    file_ = open(image_path, "rb")
    img_bytes = file_.read()
    image_str = base64.b64encode(img_bytes).decode("utf-8")
    file_.close()
    base64_src = f"data:image/jpg;base64,{image_str}"
    return base64_src
    
def pillow_local_file_to_base64(image: Image.Image):
    # pillow to local file
    TEMP_DIR = "."
    img_path = TEMP_DIR + "/" + str(uuid.uuid4()) + ".jpg"
    image.save(img_path, subsampling=0, quality=100)
    # local file base64 str
    base64_src = local_file_to_base64(img_path)
    return base64_src

def image_comparison(
    img1: str,
    img2: str,
    label1: str = "Cropped",
    label2: str = "Enhanced",
    width: int = 700,
    show_labels: bool = True,
    starting_position: int = 50,
    make_responsive: bool = True
):
    """Create a new juxtapose component.
    Parameters
    ----------
    img1: str, PosixPath, PIL.Image or URL
        Input image to compare
    label1: str or None
        Label for image 1
    label2: str or None
        Label for image 2
    width: int or None
        Width of the component in px
    show_labels: bool or None
        Show given labels on images
    starting_position: int or None
        Starting position of the slider as percent (0-100)
    make_responsive: bool or None
        Enable responsive mode
    Returns
    -------
    static_component: Boolean
        Returns a static component with a timeline
    """
    # TODO: call the local version of juxtapose (need to figure out how streamlit allows reading js/css files)
    # load css and javascript juxtapose library
    cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
    css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
    js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img_width, img_height = img1.size
    h_to_w = img_height / img_width
    height = (width * h_to_w) * 0.95
    #st.write(img_width, img_height)
    img1 = pillow_local_file_to_base64(img1)
    img2 = pillow_local_file_to_base64(img2)

    # write html block
    htmlcode = f"""
        {css_block}
        {js_block}
        <div id="foo"style="height: {height}; width: {width or '%100'};"></div>
        <script>
        slider = new juxtapose.JXSlider('#foo',
            [
                {{
                    src: '{img1}',
                    label: '{label1}',
                }},
                {{
                    src: '{img2}',
                    label: '{label2}',
                }}
            ],
            {{
                animate: true,
                showLabels: {'true' if show_labels else 'false'},
                showCredits: true,
                startingPosition: "{starting_position}%",
                makeResponsive: {'true' if make_responsive else 'false'},
            }});
        </script>
        """
    static_component = components.html(htmlcode, height=height, width=width)

    return static_component

#-------------------------

#urllib.request.urlopen('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', 'models/GFPGANv1.3.pth')
#urllib.request.urlopen('https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth', 'models/GFPGANCleanv1-NoCE-C2.pth')
#models/RealESRGAN_x2plus.pth

st.set_page_config(
    page_title="Streamlit Image Comparison",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="auto",
)

# set title and format
st.markdown(""" <style> .font {font-size:60px ; font-family: 'Sans-serif'; color: blue;} </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Biometric Analysis Tool for Media ANalytics</p>', unsafe_allow_html=True)
st.markdown("""<b>Super-Resolution using GFPGAN, a model for Blind Face Restoration with Generative Facial Prior.
               Frontal profile images perform best with the least of amount of artifacts. This version is currently
               using an older version of GFPGAN circa 2021. An updated version will be added soon.</b>""", unsafe_allow_html=True)        

st.sidebar.markdown(
    """
    <h2 style='text-align: left'>
    Super Resolution
    </h2>
    """,
    unsafe_allow_html=True,
)

st.write("##")

supported_filetypes = ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff']

st.subheader('Image Input')
uploaded_file = st.file_uploader("Choose an image file", type=supported_filetypes, accept_multiple_files=False)

output_folder = st.sidebar.text_input('Output Directory:', value="output", help="Output directory where super resolution results are stored.")
starting_position = st.sidebar.slider("Starting position of the slider:", min_value=0, max_value=100, value=50)
width = st.sidebar.slider("Component width:", min_value=400, max_value=1200, value=1000, step=100)

show_labels = st.sidebar.checkbox("Show labels", value=True)
make_responsive = st.sidebar.checkbox("Make responsive", value=True)
in_memory = st.sidebar.checkbox("In memory", value=True)

# copy image from RAM to disk
if uploaded_file is not None:
    imgpath = os.path.abspath(output_folder) + '/super_resolution/'
    
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    original_image = imgpath + uploaded_file.name

    with open(original_image, "wb") as f:
        f.write(uploaded_file.read())

    super_resolution(original_image, output_image=imgpath)
    
    static_component = image_comparison(
        img1=imgpath + "/cropped.jpg",
        img2=imgpath + "/restored.jpg",
        label1="Original (Cropped)",
        label2="Super-Resolution (GFPGAN)",
        width=width,
        starting_position=starting_position,
        show_labels=show_labels,
        make_responsive=make_responsive
    )
