import os
import sys
import shutil
import pandas as pd
import numpy as np
import exiftool
import folium
import streamlit as st
import base64
import glob
import locationtagger
import warnings
import geopandas as gpd
import tika

from tika import parser
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from PIL import Image
from stqdm import stqdm
from folium.plugins import Draw
from folium import IFrame
from streamlit_folium import st_folium, folium_static
from branca.element import Figure
from st_aggrid import AgGrid, GridOptionsBuilder #, JsCode, GridUpdateMode

warnings.simplefilter(action='ignore', category=FutureWarning)

#tika.TikaClientOnly = True
tika.initVM()

class GEOTRAX(object):
    """
    conda install -c conda-forge exiftool
    pip install PyExifTool
    
    EXIF Tags:
        https://exiftool.org/TagNames/EXIF.html

    Composite Tags:
        https://exiftool.org/TagNames/Composite.html
        
    EXIF Version 2.32:
        https://web.archive.org/web/20190624045241if_/http://www.cipa.jp:80/std/documents/e/DC-008-Translation-2019-E.pdf
        
    ExifTool Explained:
        https://adamtheautomator.com/exiftool/
        
    PhotoLinker:
        https://www.earlyinnovations.com/photolinker/metadata-tags.html
        
    Ideas:
    (1) Extract GPS from image files and plot on map
    (2) Extract locations from text and other document formats using NLP, decode to Cartesian Coordinates, plot
    (3) Extract locations from text and documents in multiple languages using NLP, decode to Cartesian Coordinates, plot
    """
    def __init__(self, output="./results"):
        """
        Constructor
        """
        st.set_page_config(page_title="Geolocating Photos Application",
                           page_icon=":warning:",
                           layout="wide",
                           initial_sidebar_state="expanded")
                           
        self.output_folder = output
        self.exiftable = pd.DataFrame()
        
        self.loctable = pd.DataFrame(columns=['location', 'latitude', 'longitude', 'source'])

        # plot GPS locations using Folium
        tiles = ['cartodbpositron',
                 'Stamen Toner',
                 'OpenStreetMap',
                 'Stamen Terrain',
                 'mapquestopen', #
                 'MapQuest Open Aerial',#
                 'Mapbox Bright', #
                 'Mapbox Control Room', #
                 'stamenwatercolor',
                 'cartodbdark_matter']
                 
        self.map = folium.Map(location=[44, -73],
                              width='68%',
                              height='100%',
                              zoom_start=4,
                              zoom_control=True,
                              scrollWheelZoom=True,
                              control_scale=True,
                              tiles='openstreetmap')

        # add map tile options
        folium.TileLayer('cartodbdark_matter').add_to(self.map)
        folium.TileLayer('cartodbpositron').add_to(self.map)
        folium.TileLayer('openstreetmap').add_to(self.map)
        folium.TileLayer('Stamen Toner').add_to(self.map)
        folium.TileLayer('Stamen Terrain').add_to(self.map)
        folium.LayerControl().add_to(self.map)

        # adds annotation tools to map
        Draw(export=True,
             position='topleft'
        ).add_to(self.map)

        #self.icon = folium.Icon(color="blue", icon="ok")
        
        #self.geolocator = Nominatim(user_agent='geotrax')
        self.geolocator = Nominatim(user_agent='my-geotrax')
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1, return_value_on_exception=None)
        
        #locator = geopy.geocoders.Nominatim(user_agent='geotrax')
        #self.geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
        
        self.figure = Figure(width=2048, height=1024)
        self.figure.add_child(self.map)

    def extract_text(self, file):
        """
        Read text from pdf/Word documents.
        """
        types = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.tsv']
        
        text = None

        if os.path.splitext(file)[1] in types:
            raw = parser.from_file(file)
            #st.write(raw)
            text = raw['content']
            dict2 = raw['metadata']
      
        return text

    def copy_images(self):
        """
        Copy uploaded images to a subfolder.
        """
        self.imgpath = os.path.abspath(self.output_folder) + '/'
        
        if os.path.exists(self.imgpath):
            shutil.rmtree(self.imgpath)
            os.makedirs(self.imgpath)
        else:
            os.makedirs(self.imgpath)

        image_list = []

        for uploaded_file in self.uploaded_files:
            image_name = self.imgpath + uploaded_file.name
            image_list.append(image_name)

            with open(image_name, "wb") as f:
                f.write(uploaded_file.read())
                
        del self.uploaded_files
        
        return image_list

    def run(self):
        """
        """
        st.title('GEOTRAX')
        
        # Google-flavored Markdown (GFM)
        # https://github.github.com/gfm/
        st.markdown('##### GEOTRAX is a simple geolocation application to plot GPS coordinates on a map extracted from image EXIF metadata.<br><br>For images that do not contain EXIF metadata, or EXIF metadata that does not contain GPS coordinates, only the header information will be displayed.<br><br>For document files (e.g. pdf, pptx, docx, xlsx, txt, etc.), location tagging is used to tag known locations from document text and plot on a map. Locations, such as addresses or cities are geocoded to retrieve coordinate values, such as extracting a city name from a document; and coordinate values (i.e. latitude and longitude values) are reverse geocoded to get a location name, such as extracting GPS coordinates from image files.<br><br><font color="blue">Image markers on the map are colored blue with an embedded camera icon</font> and <font color="green">document markers are colored green with an embedded document-type icon</font> for quick reference.', unsafe_allow_html=True)
        
#        form = st.sidebar.form('geotrax')
#        form.title("Geolocation Application")
#        form.markdown("Upload an Image and Get Its location on a Map")
#        self.uploaded_files = form.file_uploader(label="Choose a file", accept_multiple_files=True)
#        form.form_submit_button("Submit")
        
        st.sidebar.title("Geolocation Application")
        st.sidebar.markdown("Upload an Image and Get Its location on a Map")
        self.uploaded_files = st.sidebar.file_uploader(label="Choose a file", accept_multiple_files=True)

        if self.uploaded_files != []:
            self.filelist = self.copy_images()
            
            # TODO: Check for empty/0KB file size and remove from list
        
            # extract EXIF data from images
            with exiftool.ExifToolHelper() as et:
                try:
                    metadata = et.get_metadata(self.filelist)
                    #st.write(metadata)

                    # load EXIF data into dataframe
                    self.exiftable = pd.json_normalize(metadata)

                    # generate table for EXIF data
                    self.exiftable.replace(np.nan, '', regex=True, inplace=True)
                    st.subheader('EXIF Metadata')
                    gb = GridOptionsBuilder.from_dataframe(self.exiftable)
                    AgGrid(self.exiftable,
                           editable=True,
                           gridOptions=gb.build(),
                           #theme="blue", #["streamlit", "light", "dark", "blue", "fresh", "material"]
                           fit_columns_on_grid_load=True)
                    #st.write(self.exiftable.astype(str))

                except Exception as e:
                    st.markdown(f'<font color="red">ERROR: {type(e).__name__}, {e} - One or more files are empty!</font>', unsafe_allow_html=True)

        # extract locations from EXIF table
        if not self.exiftable.empty:

            # generate folium markers for all locations with GPS data
            #for index, location_info in self.exiftable.iterrows():
            for index in stqdm(range(len(self.exiftable)),
                               st_container=st.sidebar,
                               leave=True,
                               desc='Location Tagging: ',
                               gui=True):
                location_info = self.exiftable.iloc[index]
                #st.write(location_info)

                # get file extension
                _, ext = os.path.splitext(location_info.SourceFile)
                ext = ext.lower()
                
                # check if image file or document
                if ext in ['.jpg', '.png']:
                    # TODO: Make this a function

                    #st.write(f'{index} - {location_info.SourceFile} is an image')
                    
                    # generate thumbnail for popup
                    with Image.open(location_info.SourceFile) as image:
                        image.thumbnail((400, 200))
                        image_name = os.path.splitext(location_info.SourceFile)[0] + '_thumbnail.jpg'
                        image.save(image_name)
                        #st.write(image_name)
                        
                        # generate html used to display information in popup
                        encoded = base64.b64encode(open(image_name, 'rb').read()).decode('UTF-8')
                        #table = self.exiftable.iloc[index].to_frame(name='Metadata').to_html(classes="table table-striped table-bordered")# table-hover table-condensed table-responsive")

                        df_map = self.exiftable[['SourceFile',
                                                 'EXIF:CreateDate',
                                                 'EXIF:Make',
                                                 'EXIF:Model',
                                                 'File:FileSize',
                                                 'File:FileType',
                                                 'File:FileTypeExtension',
                                                 'File:ImageWidth',
                                                 'File:ImageHeight',
                                                 'Composite:ImageSize',
                                                 #'Composite:GPSDateTime',
                                                 'Composite:GPSAltitude',
                                                 'Composite:GPSLatitude',
                                                 'Composite:GPSLongitude',
                                                 'Composite:GPSPosition']].copy()
                                                 
                        st.write(df_map)

                        # rename column names to allow direct ingest into mapping library
                        df_map = df_map.rename(columns={'SourceFile':'source',
                                                        'EXIF:CreateDate':'create_date',
                                                        'EXIF:Make':'make',
                                                        'EXIF:Model':'model',
                                                        'File:FileSize':'file_size',
                                                        'File:FileType':'file_type',
                                                        'File:FileTypeExtension':'file_extension',
                                                        'File:ImageWidth':'image_width',
                                                        'File:ImageHeight':'image_height',
                                                        'Composite:ImageSize':'image_size',
                                                        #'Composite:GPSDateTime':'date_time',
                                                        'Composite:GPSAltitude':'altitude',
                                                        'Composite:GPSLatitude':'latitude',
                                                        'Composite:GPSLongitude':'longitude',
                                                        'Composite:GPSPosition':'position'})

                        temp = {'location': self.geolocator.reverse(df_map.iloc[index].position),
                               'latitude':df_map.iloc[index].latitude,
                               'longitude':df_map.iloc[index].longitude,
                               'source':df_map.iloc[index].source}
                        #st.write(temp)

                        self.loctable = self.loctable.append(temp, ignore_index=True)
                        #st.table(self.loctable.astype(str))
                        
                        table = df_map.iloc[index].to_frame(name='Metadata').to_html(classes="table table-striped table-bordered")# table-hover table-condensed 
                        #table = df_map.iloc[location_info.Index].to_frame(name='Metadata').to_html(classes="table table-striped table-bordered")# table-hover table-condensed 

                        html = f"""<center><img src="data:image/jpg;base64,{encoded}"></center><br><br>{table}"""
                        #st.write(table)                               

                        # populate popup with image and subset of EXIF data and add to marker
                        #iframe = folium.IFrame(html, width=500, height=800)
                        #popup = folium.Popup(iframe, max_width=600)
                        popup = folium.Popup(folium.Html(html, script=True), max_width=600)
                        icon = folium.Icon(color="blue", icon='camera', prefix='fa')
                        marker = folium.Marker([location_info["Composite:GPSLatitude"], location_info["Composite:GPSLongitude"]], popup=popup, icon=icon, tooltip=temp['location'])
                        marker.add_to(self.map)

                else:
                    # TODO: Make this a function
                    
                    #st.write(f'{index} - {location_info.SourceFile} is a document')
                    
                    # extract text from TXT documents                    
                    text = self.extract_text(location_info.SourceFile)
                    #st.write('==> ', text)

                    # extract locations from text
                    entities = locationtagger.find_locations(text=text)
                    #st.write(type(entities.cities))

                    # move to next file if list is empty
                    if not entities.cities:
                        continue
                    
                    # create dataframe and geocode found locations
                    df = pd.DataFrame(entities.cities)
                    df.rename(columns={0:'name'}, inplace=True)
                    df['file'] = location_info.SourceFile
                    df['location'] = df['name'].apply(self.geocode)
                    df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
                    #st.write(df.astype(str))
                    
                    # plot locations
                    for index, row in df.iterrows():
                        temp = {'location':row.location,
                                'latitude':row.point[0],
                                'longitude':row.point[1],
                                'source':row.file}
                        #st.write(temp)

                        self.loctable = self.loctable.append(temp, ignore_index=True)

                        table = row.to_frame(name=os.path.split(location_info.SourceFile)[1]).to_html(classes="table table-striped table-bordered")# table-hover table-condensed
                        html = f"""{table}"""
                        popup = folium.Popup(folium.Html(html, script=True), max_width=600)
                        #st.write(location.address, location.latitude, location.longitude, location.altitude)
                        #st.write(row)

                        # set icon based on document type
                        if ext == '.pdf':
                            icon = folium.Icon(color="green", icon='file-pdf-o', prefix='fa')
                        elif ext == '.docx':
                            icon = folium.Icon(color="green", icon='file-word-o', prefix='fa')
                        elif ext == '.xlsx':
                            icon = folium.Icon(color="green", icon='file-excel-o', prefix='fa')
                        elif ext == '.pptx':
                            icon = folium.Icon(color="green", icon='file-powerpoint-o', prefix='fa')
                        else:
                            icon = folium.Icon(color="green", icon='file-text-o', prefix='fa')

                        # plot locations on map
                        marker = folium.Marker([float(row.point[0]), float(row.point[1])], icon=icon, popup=popup, tooltip=row.location)
                        marker.add_to(self.map)

            st.subheader('Extracted Locations')
            AgGrid(self.loctable.astype(str), fit_columns_on_grid_load=True)
            #st.table(self.loctable.astype(str))

        # Note: folium static places export button to the right of the map. st_folium
        #       which is what should be used, continuously runs every time you click
        #       or do something in the map which gets annoying very quick. The export
        #       button is correctly at the bottom of the map. For now, use folium_static
        #       to avoid re-running of the map for every mouse action.
        folium_static(self.map, width=2048, height=1024)
        #st_folium(self.map, width=2048, height=1024)
        self.figure.add_child(self.map)

if __name__ == '__main__':

    geotrax = GEOTRAX()
    geotrax.run()
