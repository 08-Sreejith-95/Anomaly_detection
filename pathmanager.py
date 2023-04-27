from pathlib import Path
import re


fourdeeseg = '/data_rechenknecht03_2/students/kannath/ACDC/ACDC-Segmentation/fourdee_segmentations'
fourdeesegsplit = '/data_rechenknecht03_2/students/kannath/ACDC/ACDC-Segmentation/split_fourdee_segmentations'
fourdeeimagesplit = '/data_rechenknecht03_2/students/kannath/ACDC/ACDC-Segmentation/split_fourdee_images'
traingdata = '/data_rechenknecht03_2/students/kannath/ACDC/ACDC-Segmentation/training'

segsplit = Path(fourdeesegsplit)
imgsplit = Path(fourdeeimagesplit)
segraw = Path(fourdeeseg)
temp = list(segsplit.rglob('*gz'))
imgsplitfiles = sorted(list(imgsplit.rglob('*gz')))

segpatientfiles = sorted(list(segsplit.rglob('*gz')))
seg_raw_files = sorted(list(segraw.rglob('*gz')))

segpatient_split_path = []
imgpatient_split_path =[]
seg_raw_files_path = []

for patient in segpatientfiles:
    segpatient_split_path.append(str(patient))
for patient in imgsplitfiles:
    imgpatient_split_path.append(str(patient))
for patient in seg_raw_files:
    seg_raw_files_path.append(str(patient))

##Generalizing to 12 frames/patient, since we don't have enough frames for some patients compared to other patients in the pool
segment_path_trimmed = []
    
for file in segpatient_split_path:
   separated = file.split("_")
   pattern = re.compile("(t+)([0-3][0-9])")
   _ , frame_no = pattern.match(separated[6]).groups()
   if int(frame_no) < 13:
       segment_path_trimmed.append(file)

pass
     
   
  




    