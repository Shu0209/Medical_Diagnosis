import io
import base64
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RPImage, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os
import requests
from io import BytesIO
import pydicom
import nibabel as nib 
from Bio import Entrez


Entrez.email="your_email@example.com"

# try:
#     import pydicom
#     PYDICOM_AVAILABLE=True
# except ImportError:
#     PYDICOM_AVAILABLE=False

# try:
#     import nibabel as nib 
#     NIBABEL_AVAILABLE=True
# except ImportError:
#     NIBABEL_AVAILABLE=False


# try:
#     from Bio import Entrez
#     Entrez.email ="your_email@example.com"
#     BIOPYTHON_AVAILABLE=True
# except ImportError:
#     BIOPYTHON_AVAILABLE=False 


# Upload Document
def process_file(uploaded_file):
    ext=uploaded_file.name.split('.')[-1].lower()

    if ext in ['jpg','jpeg','png']:
        image=Image.open(uploaded_file).convert('RGB')
        return {"type":"image","data":image,"array":np.array(image)}
    
    elif ext=='dcm':
        dicom=pydicom.dcmread(uploaded_file)
        img_array=dicom.pixel_array
        img_array=((img_array-img_array.min())/(img_array.max()-img_array.min())*255).astype(np.uint8)
        

        return{
                    "type":"dicom",
                    "data":Image.fromarray(img_array),
                    "array":img_array,
                }
    elif ext in ['nii','nii.gz']:
        temp_path=f"temp_{uuid.uuid4()}.nii.gz"
        with open(temp_path,'wb') as f:
            f.write(uploaded_file.getvalue())
        nii_img=nib.load(temp_path)
        img_array=nii_img.get_fdata()[:,:,nii_img.shape[2]//2]
        img_array=((img_array-img_array.min())/(img_array.max()-img_array.min())*255).astype(np.uint8)
        os.remove(temp_path)
        return {"type":"nifti","data":Image.fromarray(img_array),"array":img_array}
    
# Heatmap
def generate_heatmap(image_array):
    if len(image_array.shape)==3:
        gray_image=cv2.cvtColor(image_array,cv2.COLOR_RGB2GRAY)
    else:
        gray_image=image_array
    
    heatmap=cv2.applyColorMap(gray_image,cv2.COLORMAP_JET)

    if len(image_array.shape)==2:
        image_array=cv2.cvtColor(image_array,cv2.COLOR_GRAY2RGB)
    overlay=cv2.addWeighted(heatmap,0.5,image_array,0.5,0)

    return Image.fromarray(overlay),Image.fromarray(heatmap)


def extract_findings_and_keywords(analysis_text):
    findings=[]
    keywords=[]

    if 'Impression:' in analysis_text:
        impression_section=analysis_text.split("Impression:")[1].strip()
        numbered_items=impression_section.split("\n")
        for item in numbered_items:
            item=item.strip()
            if item and (item[0].isdigit() or item[0]=='-' or item[0]=='*'):
                clean_item=item
                if item[0].isdigit() and "." in item[:3]:
                    clean_item=item.split(".",1)[1].strip()
                elif item[0] in ['-','*']:
                    clean_item=item[1:].strip()

                findings.append(clean_item)
                for word in clean_item.split():
                    word=word.lower().strip(',.:;()')
                    if len(word)>4 and word not in ['about','with','that','this','these','those']:
                        keywords.append(word)

    common_terms=["pneumonia","infiltrates","opacities","nodule","mass","tumor","cardiomegaly","effusion","consolidation","atelectasis","edema","fracture","fibrosis","emphysema","pneumothorax","metastasis"]

    for term in common_terms:
        if term in analysis_text.lower() and term not in keywords:
            keywords.append(term)

    keywords=list(dict.fromkeys(keywords))

    return findings, keywords[:5]

# Analysis




        




