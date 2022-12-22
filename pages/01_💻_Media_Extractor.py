# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:34:55 2022

@author: BRL
"""
import os
import shutil
import glob
import fitz
import streamlit as st
import pandas as pd
import magic
import datetime
import cv2
import pyexiv2 as pex
import configparser
import warnings
import torch

from stqdm import stqdm
from st_aggrid import AgGrid #, GridOptionsBuilder, JsCode, GridUpdateMode
from zipfile import ZipFile
from PIL import Image as Img
from mtcnn import MTCNN
from utils import crop_face

# https://blog.streamlit.io/3-steps-to-fix-app-memory-leaks/
#from memory_profiler import profile
#fp = open('safe/memory_profiler_01ME.log', 'w+')

# supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class MediaExtractor(object):
    """
    """
#    @profile(stream=fp)
    def __init__(self):
        # supported file types
        self.supported_filetypes = [
            'docx', 'docm', 'dotx', 'dotm', 'xlsx', 'xlsm', 'xltx', 'xltm',
            'pptx', 'pptm', 'potm', 'potx', 'ppsx', 'ppsm', 'odt',  'ott',
            'ods',  'ots',  'odp',  'otp',  'odg',  'doc',  'dot',  'ppt',
            'pot',  'xls',  'xlt',  'pdf', 'zip', 'mp4', 'avi', 'webm', 'wmv',
            'jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff'
        ]

        # document table
        self.extract_df = pd.DataFrame(columns=['File', 'Type', 'Size', 'Count'])

        # media table
        self.media_df = pd.DataFrame(columns=['Media', 'EXIF', 'Size', 'Height', 'Width', 'Format', 'Mode', 'Hash'])        
        
        # image table
        self.image_df = pd.DataFrame(columns=['Image', 'BoxXY', 'Height', 'Width', 'Left Eye', 'Right Eye', 'Nose', 'Mouth Left', 'Mouth Right', 'IPD', 'Confidence', 'Media', 'Hash'])

        # determine file type
        self.mime = magic.Magic(mime=True, uncompress=True)
        
        # determine device to use for face detection
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # MTCNN is used for face detection
        self.detector = MTCNN() # uses mtcnn, not dface version
        #self.detector = MTCNN(self.device) # uses the dface version of mtcnn


#    @profile(stream=fp)
    def not_extract(self, file):
        """
        Unsupported file type
        """
        file_name, file_ext = os.path.splitext(file.name)
        file_type = self.mime.from_buffer(file.read())
        file.seek(0,0)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)

        metadata = {
            'File': file_name,
            'Type': file_type,
            'Size': file_size,
            'Count': 0
        }
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        if file_ext in ['.doc', '.dot']:
            st.info(f"{file.name} is an older file format (Word 1997-2003). Please convert to docx or pdf from Microsoft Word.")

        elif file_ext in ['.ppt', '.pot']:
            st.info(f"{file.name} is an older file format (PowerPoint 1997-2003). Please convert to pptx or pdf from Microsoft PowerPoint.")
                
        elif file_ext in ['.xls', '.xlt']:
            st.info(f"{file.name} is an older file format (Excel 1997-2003). Please convert to xlsx or pdf from Microsoft Excel.")
            
        else:
            st.info(f"{file.name} is not a supported file type.")
                
#    @profile(stream=fp)
    def mso_extract(self, file, location):
        """
        Extract Microsoft Office documents from 2004 to present. Current
        format is a zip file containing various subfolders.
        """
        # TODO: make a utility function
        # read enough of the data to determine the mime typ of the file
        #print(file.read(2048))
        #file.seek(0,0)
        buffer = file.read(2048)
        file_type = self.mime.from_buffer(buffer)
        file.seek(0,0)
        
        # TODO: make a utility function
        # determine the size of the file read
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)

        file_name = file.name
        base_name = os.path.splitext(file_name)[0]
        #file_size = sys.getsizeof(file)

        with ZipFile(file) as thezip:
            #st.write(thezip.infolist())
            thezip.extractall(path=location)

        # TODO: consolidate into a single utility function
        # rename images, move images, remove directories
        if 'doc' in file.name:
            src = location + '/word/media/'
            root = os.path.splitext(file.name)[0]
            subfolder = os.path.splitext(file.name)[0] + '/extracted/'
            dest = os.path.abspath(location) + '/' + subfolder
            if os.path.exists(dest):
                shutil.rmtree(dest)
                os.makedirs(dest)
            else:
                os.makedirs(dest)               
            files = os.listdir(src)
            for f in files:
                root, ext = os.path.splitext(f)
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")
                f_new = "{}-{}-{}{}".format(base_name, root, timestamp, ext)
                os.rename(src + f, src + f_new)
                shutil.move(src + f_new, dest)
            shutil.rmtree(location + '/word')
            metadata = {'File': file_name,
                        'Type': file_type,
                        'Size': file_size,
                        'Count': len(files)}
            self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        elif 'ppt' in file.name:
            src = location + '/ppt/media/'
            root = os.path.splitext(file.name)[0]
            subfolder = os.path.splitext(file.name)[0] + '/extracted/'
            dest = os.path.abspath(location) + '/' + subfolder
            if os.path.exists(dest):
                shutil.rmtree(dest)
                os.makedirs(dest)
            else:
                os.makedirs(dest)
            files = os.listdir(src)
            for f in files:
                root, ext = os.path.splitext(f)
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")
                f_new = "{}-{}-{}{}".format(base_name, root, timestamp, ext)
                os.rename(src + f, src + f_new)
                shutil.move(src + f_new, dest)
            shutil.rmtree(location + '/ppt')
            metadata = {'File': file_name,
                        'Type': file_type,
                        'Size': file_size,
                        'Count': len(files)}
            self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        elif 'xl' in file.name:
            src = location + '/xl/media/'
            root = os.path.splitext(file.name)[0]
            subfolder = os.path.splitext(file.name)[0] + '/extracted/'
            dest = os.path.abspath(location) + '/' + subfolder
            if os.path.exists(dest):
                shutil.rmtree(dest)
                os.makedirs(dest)
            else:
                os.makedirs(dest)
            files = os.listdir(src)
            for f in files:
                root, ext = os.path.splitext(f)
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")
                f_new = "{}-{}-{}{}".format(base_name, root, timestamp, ext)
                os.rename(src + f, src + f_new)
                shutil.move(src + f_new, dest)
            shutil.rmtree(location + '/xl')
            metadata = {'File': file_name,
                        'Type': file_type,
                        'Size': file_size,
                        'Count': len(files)}
            self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        shutil.rmtree(location + '/_rels')
        shutil.rmtree(location + '/docProps')
        [os.remove(f) for f in glob.glob(location + '/*.xml')]

        return dest

#    @profile(stream=fp)
    def zip_extract(self, file, location):
        """
        https://docs.python.org/3/library/zipfile.html#module-zipfile
        """
        # get file stats
        file_name = os.path.basename(file.name)
        #file_type = self.mime.from_buffer(file.read())
        file.seek(0,0)
        
        # determine the size of the file read
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)
        
        with ZipFile(file) as thezip:
            #st.write(thezip.infolist())
            for i in stqdm(range(len(thezip.infolist())),
                           #st_container=st.sidebar,
                           leave=True,
                           desc='ZIP Extraction: ',
                           gui=True):
            #for zipinfo in thezip.infolist():
                zipinfo = thezip.filelist[i]
                file_type = os.path.splitext(zipinfo.filename)[1][1:]

                #st.write(zipinfo.filename)

                if file_type in ['doc', 'dot']:
                    st.info(f"{zipinfo.filename} is an older file format (Word 1997-2003). Please convert to docx or pdf from Microsoft Word. Open document then click save as.")
    
                elif file_type in ['ppt', 'pot']:
                    st.info(f"{zipinfo.filename} is an older file format (PowerPoint 1997-2003). Please convert to pptx or pdf from Microsoft PowerPoint. Open document then click save as.")
    
                elif file_type in ['xls', 'xlt']:
                    st.info(f"{zipinfo.filename} is an older file format (Excel 1997-2003). Please convert to xlsx or pdf from Microsoft Excel. Open document then click save as.")

                elif file_type in ['pdf']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.pdf_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                elif file_type in ['docx', 'docm', 'dotm', 'dotx', 'xlsx',
                                   'xlsb', 'xlsm', 'xltm', 'xltx', 'potx',
                                   'ppsm', 'ppsx', 'pptm', 'pptx', 'potm']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.mso_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                elif file_type in ['mp4', 'webm', 'avi', 'wmv']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.vid_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                    
                elif file_type in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff']:
                    with thezip.open(zipinfo) as thefile:
                        imgpath = self.img_extract(thefile, self.output_folder)
                        self.__get_media(imgpath)
                        self.__get_images(imgpath)

                else:
                    pass
                    #st.info(f"Ignoring {zipinfo.filename} as it is not a supported file_type")

        metadata = {'File': file_name, 'Type': 'application/zip', 'Size': file_size, 'Count': len(thezip.infolist())}
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

#    @profile(stream=fp)
    def pdf_extract(self, file, location):
        """
        https://pymupdf.readthedocs.io/en/latest/index.html
        """
        # read enough of the data to determine the mime typ of the file
        file_type = self.mime.from_buffer(file.read(2048))
        file.seek(0,0)
        
        # determine the size of the file read
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)

        # open pdf file
        pdf_file = fitz.open('pdf', file.read())
        #st.write(pdf_file.metadata)
        #st.write(file)

        # if self.media_tag == "":
        #     root, ext = os.path.splitext(file.name)
        # else:
        #     root = self.media_tag
        root, ext = os.path.splitext(file.name)
        
        #st.write(root)

        subfolder = os.path.splitext(file.name)[0] + '/extracted/'
        document_path = os.path.abspath(location) + '/' + subfolder
        if os.path.exists(document_path):
            shutil.rmtree(document_path)
            os.makedirs(document_path)
        else:
            os.makedirs(document_path)

        # # create directory if it does not exist
        # if not os.path.exists(location):
        #     os.mkdir(location)

        # image counter
        nimags = 0

        # iterating through each page in the pdf
        for current_page_index in range(pdf_file.page_count):

            #iterating through each image in every page of PDF
            #for img_index, img in enumerate(pdf_file.getPageImageList(current_page_index)):
            for img_index, img in enumerate(pdf_file.get_page_images(current_page_index)):
                  xref = img[0]
                  image = fitz.Pixmap(pdf_file, xref)
    
                  #if it is a is GRAY or RGB image
                  if image.n < 5:        
                      #image.writePNG("{}/{}_pg{}-idx{}.png".format(location, os.path.splitext(file.name)[0], current_page_index, img_index))
                      #image.writePNG("{}/{}-{}.png".format(document_path, root, datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")))
                      image.save("{}/{}-{}.png".format(document_path, root, datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")))

                  #if it is CMYK: convert to RGB first
                  else:                
                      new_image = fitz.Pixmap(fitz.csRGB, image)
                      #new_image.writePNG("{}/{}_pg{}-idx{}.png".format(location, os.path.splitext(file.name)[0], current_page_index, img_index))
                      new_image.writePNG("{}/{}-{}-{}.png".format(document_path, root, img_index, datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")))
                      
                  nimags = nimags + 1

        try:
            metadata = {'File': file.name, 'Type': file.type, 'Size': file.size, 'Count': nimags}
        except:
            metadata = {'File': file.name, 'Type': file_type, 'Size': file_size, 'Count': nimags}

        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        # if not self.extract_df.empty:
        #     st.subheader("Documents")
        #     #st.dataframe(self.extract_df, width=None, height=None)
        #     AgGrid(self.extract_df, fit_columns_on_grid_load=True)
        
        return document_path

#    @profile(stream=fp)
    def vid_extract(self, file, location):
        """
        """
        # create extraction folder
        subfolder = os.path.splitext(file.name)[0] + '_video/extracted/'
        video_path = os.path.abspath(location) + '/' + subfolder
        if os.path.exists(video_path):
            shutil.rmtree(video_path)
            os.makedirs(video_path)
        else:
            os.makedirs(video_path)

        # copy buffer to output folder
        video_file = os.path.abspath(self.output_folder) + '/' + file.name        
        #st.write(video_file)
        with open(video_file, "wb") as f:
            f.write(file.read())

        # get file stats
        file_name = os.path.basename(video_file)
        file_type = self.mime.from_file(video_file)
        file_size = os.path.getsize(video_file)

        # initialize video frame capture        
        vidcap = cv2.VideoCapture(video_file)
        success, image = vidcap.read()
        count = 0
        max_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # extract frames from video as jpg
        for i in stqdm(range(max_frames),
                       #st_container=st.sidebar,
                       leave=True,
                       desc='Video Extraction: ',
                       gui=True):

            # break from loop is frame extraction fails
            if not success:
                break

            # adjust skip frames to avoid division by zero
            #   skip_frames = 0, grab every frame (no frames skipped)
            #   skip_frames = 1, grab every 2nd frame (grab, skip, grab, skip, grab,...)
            #   skip_frames = 2, grab every 3rd frame (grab, skip, skip, grab, skip, skip, grab,...)
            #   etc.
            skip = self.skip_frames + 1
            
            # write image to output path
            if i % skip == 0:
                cv2.imwrite(video_path + os.path.splitext(file_name)[0] + f"_frame_{i}-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f") + ".jpg", image) # save frame as JPEG file      
                count += 1
            success, image = vidcap.read()

        # write file and media stats to dataframe
        metadata = {'File': file_name, 'Type': file_type, 'Size': file_size, 'Count': count}
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        vidcap.release()
        os.remove(video_file)

        return video_path

#    @profile(stream=fp)
    def img_extract(self, file, location):
        """
        """
        file_name, ext = os.path.splitext(os.path.basename(file.name))
        file_type = self.mime.from_buffer(file.read(2048))
        file.seek(0,0)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0,0)

        subfolder = os.path.splitext(file_name)[0] + '/extracted/'
        imgpath = os.path.abspath(location) + '/' + subfolder
        if os.path.exists(imgpath):
            shutil.rmtree(imgpath)
            os.makedirs(imgpath)
        else:
            os.makedirs(imgpath) 

        metadata = {
            'File': file_name,
            'Type': file_type,
            'Size': file_size,
            'Count': 1
        }
        self.extract_df = self.extract_df.append(metadata, ignore_index=True)

        with open(imgpath + file_name + '-' + datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f") + ext, "wb") as f:
            f.write(file.read())
            
        return imgpath

#    @profile(stream=fp)
    def __get_media(self, output_folder):
        """
        """
        try:
            files = os.listdir(output_folder)

            for f in files:
                imgfile = output_folder + '/' + f
                try:
                    im = Img.open(imgfile)
                except Exception as e:
                    st.error(e)
                    break

                # check for EXIV data
                try:
                    pimg = pex.Image(imgfile)
                    data = pimg.read_exif()
                    pimg.close()

                    if data:
                        exif_data = "yes"
                        # st.info(data)
                    else:
                        exif_data = "no"

                except Exception as e:
                    st.error(e)
                    break

                cropped_hash = cv2.img_hash.averageHash(cv2.imread(imgfile))[0]
                cropped_hash = ''.join(hex(i)[2:] for i in cropped_hash)

                # using imagehash library instead of cv2.img_hash because we opened the file with PIL library
                # TODO: Should we use one or the other, or does it really matter which is used?
                metadata = {'Media': f,
                            'EXIF': exif_data,
                            'Size': os.path.getsize(output_folder + '/' + f),
                            'Height': im.height,
                            'Width': im.width,
                            'Format': im.format,
                            'Mode': im.mode,
                            #'Hash': imagehash.average_hash(im)}
                            'Hash': cropped_hash}
                self.media_df = self.media_df.append(metadata, ignore_index=True)
        except:
            pass

#    @profile(stream=fp)
    def __get_images(self, output_folder):
        """
        The detector returns a list of JSON objects. Each JSON object contains
        three main keys:
        - 'box' is formatted as [x, y, width, height]
        - 'confidence' is the probability for a bounding box to be matching a face
        - 'keypoints' are formatted into a JSON object with the keys:
            * 'left_eye',
            * 'right_eye',
            * 'nose',
            * 'mouth_left',
            * 'mouth_right'
          Each keypoint is identified by a pixel position (x, y).
        """
        media_files = glob.glob(output_folder + '*.*')

        face_count = 0        
        max_files = len(media_files)

        # set image path
        output_folder = output_folder.split('extracted')[0]
        image_path = output_folder + 'cropped/'
        detection_path = output_folder + 'detected/'

        if os.path.exists(image_path):
            shutil.rmtree(image_path)
            os.mkdir(image_path)
            shutil.rmtree(detection_path)
            os.mkdir(detection_path)
        else:
            os.mkdir(image_path)
            os.mkdir(detection_path)

        for i in stqdm(range(max_files),
                        #st_container=st.sidebar,
                        leave=True,
                        desc='Face Detection: ',
                        gui=True):

            f = media_files[i]
            
            try:
                image = cv2.imread(f)
                detection_image = image.copy()
                bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detections = self.detector.detect_faces(bgr)
                
                # filtering detections with confidence greater than confidence threshold
                for idx, det in enumerate(detections):
                    if det['confidence'] >= self.confidence:
                        face_count = face_count + 1
                        x, y, width, height = det['box']
                        keypoints = det['keypoints']

                        # calculate ipd using euclidean distance
                        #left_eye = Point(keypoints['left_eye'][0], keypoints['left_eye'][1])
                        #right_eye = Point(keypoints['right_eye'][0], keypoints['right_eye'][1])
                        #ipd = Line(left_eye, right_eye)
                        
                        # calculate ipd by taking the delta between the eyes (pixel distance)
                        #  this is roughly equivalent to calculating ipd using euclidean distance (above)
                        ipd = keypoints['right_eye'][0] - keypoints['left_eye'][0]

                        # draw bounding box for face; and points for eyes, nose and mouth
                        cv2.rectangle(detection_image, (x,y), (x+width,y+height), (0,155,255), 1)
                        #cv2.circle(img_with_dets, (keypoints['left_eye']), 2, (0,155,255), 1)
                        #cv2.circle(img_with_dets, (keypoints['right_eye']), 2, (0,155,255), 1)
                        #cv2.circle(img_with_dets, (keypoints['nose']), 2, (0,155,255), 1)
                        #cv2.circle(img_with_dets, (keypoints['mouth_left']), 2, (0,155,255), 1)
                        #cv2.circle(img_with_dets, (keypoints['mouth_right']), 2, (0,155,255), 1)

                        # crop the image by increasing the detection bounding box (i.e. margin)
                        #  a margin of 1 is the detection bounding box
                        box = x, y, x+width, y+height
                        cropped_image = crop_face(image, box, margin=1.65)

                        # creates clearer picture by changing pixel intensity and increasing overall contrast
                        #  ala part of image quality (probably need to come up with a metric here)
                        if self.normalize:
                            cropped_image = cv2.normalize(cropped_image, None, 0, 255, cv2.NORM_MINMAX)
                        
                        # set cropped images to have the same height and width
                        #   we will want this image size to be consistent for all cropped images to be clustered
                        cropped_image = cv2.resize(cropped_image, (self.image_width, self.image_height))
                        
                        # export image to disk as a PNG file
                        if self.image_format == 'PNG':
                            cropped_image_name = image_path + os.path.splitext(os.path.basename(f))[0] + '-' + f'image_{idx}' + '.png'
                        elif self.image_format == 'JPG':
                            cropped_image_name = image_path + os.path.splitext(os.path.basename(f))[0] + '-' + f'image_{idx}' + '.jpg'

                        cv2.imwrite(cropped_image_name, cropped_image)

                        # convert cropped image into an image hash using cv2
                        cropped_hash = cv2.img_hash.averageHash(cropped_image)[0]
                        cropped_hash = ''.join(hex(i)[2:] for i in cropped_hash)

                        metadata = {
                            'Image': os.path.basename(cropped_image_name),
                            'BoxXY': (x, y),
                            'Height': height,
                            'Width': width,
                            'Left Eye': keypoints['left_eye'],
                            'Right Eye': keypoints['right_eye'],
                            'Nose': keypoints['nose'],
                            'Mouth Left': keypoints['mouth_left'],
                            'Mouth Right': keypoints['mouth_right'],
                            'IPD': ipd,
                            'Confidence': det['confidence'],
                            'Media': os.path.basename(f),
                            'Hash': cropped_hash
                        }
                        
                        self.image_df = self.image_df.append(metadata, ignore_index=True)

                detection_image_name = detection_path + os.path.splitext(os.path.basename(f))[0] + '.png'
                cv2.imwrite(detection_image_name, detection_image)

            except Exception as e:
                st.error(e)
                
#    @profile(stream=fp)
    def run(self):
        """
        """
        # set streamlit page defaults
        st.set_page_config(
            layout = 'wide', # centered, wide, dashboard
            initial_sidebar_state = 'auto', # auto, expanded, collapsed
            page_title = 'Media Extractor',
            page_icon = Img.open(r"./assets/baticon.png") #':eyes:' # https://emojipedia.org/shortcodes/
        )
        
        #sidebar settings
        st.sidebar.subheader('Media Extractor Settings')
        self.output_folder = st.sidebar.text_input('Output Directory:', value=os.path.abspath(r"output"), key='media_output', help="Output directory where extracted, detected, and cropped images are stored.")
        self.confidence = float(st.sidebar.slider('Confidence:',
                                            min_value=0.0,
                                            max_value=1.0,
                                            step=0.01,
                                            value=0.90,
                                            #on_change=self.__update_settings,
                                            #args=(self.config_file, 'MEDIA_EXTRACTOR', 'Confidence', str(self.confidence)),
                                            help="Face detection confidence value."))
        self.skip_frames = st.sidebar.slider('Skip Frames:', min_value=0, max_value=300, step=1, value=30, key='media_skip', help="The number of video frames to skip before saving image. (Note: skip frames/30 frames per sec = number of seconds skipped)")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            self.image_height = int(st.text_input('Image Height:', value=380, help="Cropped image height."))
        with col2:
            self.image_width = int(st.text_input('Image Width:', value=380, help="Cropped image width."))

        self.image_format = st.sidebar.selectbox('Image Format:', ('JPG', 'PNG'), index=0, help='Cropped image format.')
        
        # col1a, col2a = st.sidebar.columns(2)
        # with col1a:
        #     self.normalize = st.checkbox("Normalize", value=True, help="Creates clearer picture by changing pixel intensity and increasing overall contrast.")
        # with col2a:
        #     st.button("💾 Save", on_click=self.__update_settings)

        self.normalize = st.sidebar.checkbox("Normalize", value=True, help="Creates clearer picture by changing pixel intensity and increasing overall contrast.")
            
        st.sidebar.markdown("---")
        
        # set title and format
        st.markdown(""" <style> .font {font-size:60px ; font-family: 'Sans-serif'; color: blue;} </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Biometric Analysis Tool for Media ANalytics</p>', unsafe_allow_html=True)

        st.subheader('Media Input')
        self.uploaded_files = st.file_uploader("Choose a media file (image, video, or document)", type=self.supported_filetypes, accept_multiple_files=True)

#        with st.form("my-form", clear_on_submit=False):
#            self.uploaded_files = st.file_uploader("Choose a media file (image, video, or document)", type=self.supported_filetypes, accept_multiple_files=True)
#            submitted = st.form_submit_button("UPLOAD!")
            #st.write(submitted, self.uploaded_files)
                        
##            if submitted and self.uploaded_files is not None:
#            if submitted and self.uploaded_files != []:
        if self.uploaded_files != []:
            max_files = len(self.uploaded_files)
            #st.write(f'max_files: {max_files}')

            for i in stqdm(range(max_files),
                            #st_container=st.sidebar,
                            leave=True,
                            desc='Media Extraction: ',
                            gui=True):

                uploaded_file = self.uploaded_files[i] 
                st.sidebar.write(f"Processing File: {uploaded_file.name}")

                # split filename to get extension and remove the '.'
                file_type = os.path.splitext(uploaded_file.name)[1][1:]

                if file_type in ['doc', 'dot']:
                    self.not_extract(uploaded_file)

                elif file_type in ['ppt', 'pot']:
                    self.not_extract(uploaded_file)
                    
                elif file_type in ['xls', 'xlt']:
                    self.not_extract(uploaded_file)

                elif file_type in ['pdf']:
                    imgpath = self.pdf_extract(uploaded_file, self.output_folder)
                    self.__get_media(imgpath)
                    self.__get_images(imgpath)

                elif file_type in ['zip']:
                    self.zip_extract(uploaded_file, self.output_folder)
                    
                elif file_type in ['docx', 'docm', 'dotm', 'dotx', 'xlsx',
                                   'xlsb', 'xlsm', 'xltm', 'xltx', 'potx',
                                   'ppsm', 'ppsx', 'pptm', 'pptx', 'potm']:
                    imgpath = self.mso_extract(uploaded_file, self.output_folder)
                    self.__get_media(imgpath)
                    self.__get_images(imgpath)

                elif file_type in ['mp4', 'avi', 'webm', 'wmv']:                
                    imgpath = self.vid_extract(uploaded_file, self.output_folder)
                    self.__get_media(imgpath)
                    self.__get_images(imgpath)
                    
                elif file_type in file_type in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff']:
                    imgpath = self.img_extract(uploaded_file, self.output_folder)
                    self.__get_media(imgpath)
                    self.__get_images(imgpath)

                else:
                    self.not_extract(uploaded_file)
                    
            st.success('Process completed')
                    
        else:
            st.info('Please select files to be processed.')

        # create metadata table
        if not self.extract_df.empty:
            st.subheader("Documents")
            #st.dataframe(self.extract_df, width=None, height=None)
            AgGrid(self.extract_df, fit_columns_on_grid_load=True)
            st.info(f"* Total of {max_files} files processed, {self.extract_df['Count'].sum()} media files extracted")

        if not self.media_df.empty:
            st.subheader("Media")
            AgGrid(self.media_df, fit_columns_on_grid_load=True)

        if not self.image_df.empty:
            st.subheader("Images")
            AgGrid(self.image_df, fit_columns_on_grid_load=True)
            st.info(f"* Found a total of {len(self.image_df)} face(s) in media files")

if __name__ == '__main__':
    mex = MediaExtractor()    
    mex.run()

    #from pyinstrument import Profiler
    #mex = MediaExtractor()    
    #profiler = Profiler()
    #profiler.start()
    #mex.run()
    #profiler.stop()
    #profiler.print()

    #from pycallgraph import PyCallGraph
    #from pycallgraph import Config
    #from pycallgraph import GlobbingFilter
    #from pycallgraph.output import GraphvizOutput
    #config = Config(max_depth=3)
    #config.trace_filter = GlobbingFilter(exclude=['pycallgraph.*', 'AgGridReturn.*', '_*', '<*>'])
    #graphviz = GraphvizOutput()
    #graphviz.output_file = 'me_call_graph_t01.png'
    #mex = MediaExtractor()    
    #with PyCallGraph(output=graphviz, config=config):
    #    mex.run()

