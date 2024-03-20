# YS-A
An end-to-end automatic algal area inversion model:
+ Fisheye lens correction and inverse perspective transformation
+ Use yolov5-v6.2 to generate prompts(bbxes) and pass them to sam for alga segmentation and area inversion task
+ batch inference

# Steps:
+ Pass customed datas into fisheye_lens_correction.ipynb for fisheye lens correction and inverse perspective transformation
+ Pass corrected images into prompts generator: yolov5-v6.2
```python
cd YS-A/yolov5-6.2
pip install -r requirements.txt
python detect.py --weights YS-A/yolov5-6.2/runs/exp/last.pt --save-txt --source 0                        
                                                                         img.jpg                         # image
                                                                         screen                          # screenshot
                                                                         path/                           # directory
                                                                         'path/*.jpg'                    # glob
```
+ batch inference (replace your own corrected images path/prompts path):
```python
cd YS-A/segment-anything-main
python end-to-end.py
```
