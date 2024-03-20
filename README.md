# YS-A
An end-to-end automatic algal area inversion model:
+ Using yolov5-v6.2 as a prompt generator for sam

# Steps：
+ Fisheye lens view correction and inverse perspective transformation
+ Use yolov5-v6.2 to generate prompts(bbxes) and pass them to sam for alga segmentation and area inversion task
+ batch inference
```python
python end-to-end.py
```
