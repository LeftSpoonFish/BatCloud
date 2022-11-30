import os
import shutil
import pandas as pd
import numpy as np
import folium
import streamlit as st
import base64
import locationtagger
import warnings
import tika

from tika import parser
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from stqdm import stqdm
from folium.plugins import Draw
from streamlit_folium import st_folium, folium_static
from branca.element import Figure
from st_aggrid import AgGrid, GridOptionsBuilder
from PIL import Image

# TAGS/GPSTAGS dictionary maps 16-bit integer EXIF/GPS tag enumerations to descriptive string names
from PIL.ExifTags import TAGS     
from PIL.ExifTags import GPSTAGS

# build reverse dicts
_TAGS_r = dict(((v, k) for k, v in TAGS.items()))
_GPSTAGS_r = dict(((v, k) for k, v in GPSTAGS.items()))

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# initialize Tika engine
tika.initVM()

class GEOTRAX(object):
    """
    Description of EXIF file format:
        https://www.media.mit.edu/pia/Research/deepview/exif.html

    EXIF Tags:
        https://exiftool.org/TagNames/EXIF.html

    Composite Tags:
        https://exiftool.org/TagNames/Composite.html
        
    EXIF Version 2.32:
        https://web.archive.org/web/20190624045241if_/http://www.cipa.jp:80/std/documents/e/DC-008-Translation-2019-E.pdf

    PhotoLinker:
        https://www.earlyinnovations.com/photolinker/metadata-tags.html        
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
        Draw(export=False,
             position='topleft'
        ).add_to(self.map)

        # limit the rate to check for location to prevent server lockout
        self.geolocator = Nominatim(user_agent='geotrax')
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1, return_value_on_exception=None)
                
        # figure that houses the folium map
        self.figure = Figure(width=2048, height=1024)
        self.figure.add_child(self.map)

    def get_data(self, path):
        """
        Wrapper function to get EXIF and GPS data.
        """
        try:
            exif_data = self.get_exif(path)
        except:
            return None, None

        try:
            gps_data = self.get_gpstags(exif_data)
        except:
            return exif_data, None
        
        return exif_data, gps_data
            
    def get_exif(self, path):
        """
        Get EXIF data from image.
        """
        image = Image.open(path)
        exif = image._getexif()

        return exif

    def get_gpstags(self, exif):
        """
        Extract GPS info from EXIF data.
        """
        geotagging = {}
        gpsinfo = exif[_TAGS_r["GPSInfo"]]
        #print("\n".join([str((GPSTAGS[k], gpsinfo[k])) for k in gpsinfo.keys()]))

        # create geotagging dictionary from GPSInfo tag
        for k in gpsinfo.keys():
            geotagging[GPSTAGS[k]] = gpsinfo[k]

        return geotagging

    def get_decimal_from_dms(self, dms, ref):
        """
        Utility function to convert DMS coordinates to degrees.
        """
        degrees = dms[0]
        minutes = dms[1]/ 60.0
        seconds = dms[2] / 3600.0

        if ref in ['S', 'W']:
            degrees = -degrees
            minutes = -minutes
            seconds = -seconds

        return round(degrees + minutes + seconds, 5)

    def get_coordinates(self, geotags):
        """
        Wrapper function to convert DMS coordinates.
        """
        lat = self.get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])
        lon = self.get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])

        return (lat,lon)

    def extract_text(self, file):
        """
        Read text from pdf/Word documents.
        """
        types = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.tsv']
        
        text = None

        if os.path.splitext(file)[1] in types:
            raw = parser.from_file(file)
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
        Process image/document by extracting locations (e.g. GPS coordinates, 
        geocoded text-based locations) and plot on a map.
        """
        st.title('GEOTRAX')
        
        # Google-flavored Markdown (GFM)
        # https://github.github.com/gfm/
        st.markdown('##### GEOTRAX is a simple geolocation application to plot GPS coordinates on a map extracted from image EXIF metadata.<br><br>For images that do not contain EXIF metadata, or EXIF metadata that does not contain GPS coordinates, only the header information will be displayed.<br><br>For document files (e.g. pdf, pptx, docx, xlsx, txt, etc.), location tagging is used to tag known locations from document text and plot on a map. Locations, such as addresses or cities are geocoded to retrieve coordinate values, such as extracting a city name from a document; and coordinate values (i.e. latitude and longitude values) are reverse geocoded to get a location name, such as extracting GPS coordinates from image files.<br><br><font color="blue">Image markers on the map are colored blue with an embedded camera icon</font> and <font color="green">document markers are colored green with an embedded document-type icon</font> for quick reference.', unsafe_allow_html=True)
        
        st.sidebar.title("Geolocation Application")
        st.sidebar.markdown("Upload an Image and Get Its location on a Map")
        self.uploaded_files = st.sidebar.file_uploader(label="Choose a file", accept_multiple_files=True)

        if self.uploaded_files != []:
            self.filelist = self.copy_images()
            
            # extract EXIF data from images
            for file in (self.filelist):
                exif, gps_dict = self.get_data(file)

                # add document to EXIF Table
                if exif is None:
                    file_dict = {}
                    file_dict['SourceFile'] = file                
                    temp_df = pd.DataFrame([file_dict])
                    self.exiftable = self.exiftable.append(temp_df, ignore_index=True)
                    self.exiftable.replace(np.nan, '', regex=True, inplace=True)
                    continue

                keys = list(exif.keys())

                # remove keys containing large amount of information
                try:
                    keys.remove(_TAGS_r["MakerNote"])
                except:
                    pass

                try:
                    keys.remove(_TAGS_r["UserComment"])
                except:
                    pass

                try:
                    keys.remove(_TAGS_r["ImageDescription"])
                except:
                    pass

                try:
                    keys.remove(_TAGS_r["GPSInfo"])
                except:
                    pass

                keys = [k for k in keys if k in TAGS]

                exif_dict = {}
                for k in keys:
                    exif_dict[TAGS[k]] = exif[k]
                    
                # add source file tag
                exif_dict['SourceFile'] = file

                try:
                    # fix gps data
                    gps_data = self.get_coordinates(gps_dict)
                    gps_dict['GPSLatitude'] = gps_data[0]
                    gps_dict['GPSLongitude'] = gps_data[1]
                    gps_dict['GPSTimeStamp'] = str(int(gps_dict['GPSTimeStamp'][0])) + ':' + str(int(gps_dict['GPSTimeStamp'][1])) + ':' + str(int(gps_dict['GPSTimeStamp'][2])) 
                    gps_dict['GPSDateTime'] = gps_dict['GPSDateStamp'] + " " + gps_dict['GPSTimeStamp']
                    gps_dict['GPSPosition'] = str(gps_dict['GPSLatitude']) + " " + str(gps_dict['GPSLongitude'])
                
                    # merge exif and gps dictionaries
                    exif_data = exif_dict | gps_dict

                    # create dataframe from dictionary
                    temp_df = pd.DataFrame([exif_data])
                    self.exiftable = self.exiftable.append(temp_df, ignore_index=True)
                    self.exiftable.replace(np.nan, '', regex=True, inplace=True)

                except:
                    exif_data = exif_dict
                    temp_df = pd.DataFrame([exif_data])
                    self.exiftable = self.exiftable.append(temp_df, ignore_index=True)
                    self.exiftable.replace(np.nan, '', regex=True, inplace=True)

        st.subheader('EXIF Metadata')
        if not self.exiftable.empty:
            sourcefile = self.exiftable["SourceFile"]
            filename = [os.path.basename(f) for f in sourcefile]
            self.exiftable = self.exiftable.drop(columns=["SourceFile"])
            self.exiftable.insert(loc=0, column='SourceFile', value=sourcefile)            
            self.exiftable.insert(loc=0, column='Filename', value=filename)            
            #st.write(self.exiftable.astype(str))
            gb = GridOptionsBuilder.from_dataframe(self.exiftable)
            AgGrid(self.exiftable.astype(str),
                   editable=True,
                   gridOptions=gb.build())

        # extract locations from EXIF table
        if not self.exiftable.empty:

            # generate folium markers for all locations with GPS data
            for index in stqdm(range(len(self.exiftable)),
                               st_container=st.sidebar,
                               leave=True,
                               desc='Location Tagging: ',
                               gui=True):

                # extract row - it is now panda series
                location_info = self.exiftable.iloc[index]
                
                # get file extension
                _, ext = os.path.splitext(location_info.SourceFile)
                
                # check if image file or document
                if ext in ['.jpg', '.png']:

                    # check if GPS columns exist in metadata otherwise go to next file
                    if not "GPSLatitude" in self.exiftable or not "GPSLongitude" in self.exiftable:
                        continue
                    
                    # check if GPS metadata contains literal values otherwise go to next file
                    if location_info.GPSLatitude == "" or location_info.GPSLatitude == None:
                        continue

                    location_info = location_info[['SourceFile',
                                                   'Filename',
                                                   'Make',
                                                   'Model',
                                                   'Software',
                                                   'DateTimeOriginal',
                                                   'ExifImageWidth',
                                                   'ExifImageHeight',
                                                   'XResolution',
                                                   'YResolution',
                                                   'GPSLatitude',
                                                   'GPSLongitude',
                                                   'GPSAltitude']]

                    # geocode GPS coordinate to a physical location and add to data
                    location_info['Location'] = self.geolocator.reverse((location_info["GPSLatitude"], location_info["GPSLongitude"]))
                    
                    # create thumbnail and add popup to map 
                    with Image.open(location_info.SourceFile) as image:
                        image.thumbnail((400, 200))
                        image_name = os.path.splitext(location_info.SourceFile)[0] + '_thumbnail.jpg'
                        image.save(image_name)
                        
                        # generate html used to display information in popup
                        encoded = base64.b64encode(open(image_name, 'rb').read()).decode('UTF-8')

                        # extract GPS location
                        temp = {'location':  location_info["Location"],
                                'latitude':  location_info["GPSLatitude"],
                                'longitude': location_info["GPSLongitude"],
                                'source':    location_info.SourceFile}

                        # create HTML table to use in popup
                        table = location_info.to_frame(name='Metadata').to_html(classes="table table-striped table-bordered")
                        html = f"""<center><img src="data:image/jpg;base64,{encoded}"></center><br><br>{table}"""

                        # populate popup with image and subset of EXIF data and add to marker
                        popup = folium.Popup(folium.Html(html, script=True), max_width=600)
                        icon = folium.Icon(color="blue", icon='camera', prefix='fa')
                        marker = folium.Marker([location_info["GPSLatitude"], location_info["GPSLongitude"]], popup=popup, icon=icon, tooltip=location_info['Location'])
                        marker.add_to(self.map)

                else:
                    # extract text from TXT documents                    
                    text = self.extract_text(location_info.SourceFile)

                    # extract locations from text
                    entities = locationtagger.find_locations(text=text)

                    # move to next file if list is empty
                    if not entities.cities:
                        continue
                    
                    # create dataframe and geocode found locations
                    df = pd.DataFrame(entities.cities)
                    df.rename(columns={0:'name'}, inplace=True)
                    df['file'] = location_info.SourceFile
                    df['location'] = df['name'].apply(self.geocode)
                    df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
                    
                    # plot locations
                    for index, row in df.iterrows():
                        temp = {'location':row.location,
                                'latitude':row.point[0],
                                'longitude':row.point[1],
                                'source':row.file}

                        self.loctable = self.loctable.append(temp, ignore_index=True)

                        table = row.to_frame(name=os.path.split(location_info.SourceFile)[1]).to_html(classes="table table-striped table-bordered")
                        html = f"""{table}"""
                        popup = folium.Popup(folium.Html(html, script=True), max_width=600)

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

            self.loctable = self.loctable.rename(columns={'location':'Location',
                                                          'latitude':'Latitude',
                                                          'longitude':'Longitude',
                                                          'source':'Source'})
            if not self.loctable.empty:
                st.subheader('Extracted Locations')
                AgGrid(self.loctable.astype(str), fit_columns_on_grid_load=True)

        # Note: folium static places export button to the right of the map. st_folium
        #       which is what should be used, continuously runs every time you click
        #       or do something in the map which gets annoying very quick, but the export
        #       button is correctly at the bottom of the map. For now, use folium_static
        #       to avoid re-running of the map for every mouse action despite odd location
        #       of the export button.
        folium_static(self.map, width=1960, height=1024)
        #st_folium(self.map, width=2048, height=1024)
        self.figure.add_child(self.map)

if __name__ == '__main__':
    geotrax = GEOTRAX()
    geotrax.run()