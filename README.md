# Official YOLOv7 trained on a custom dataset for window detection Open/Closed







## Testing




On video:
``` shell
python detect.py --weights window.pt --conf 0.25 --img-size 640 --source yourvideo.mp4 or select device camera 
```

On image:
``` shell
python detect.py --weights window.pt --conf 0.25 --img-size 640 --source inference/images/image.jpg
```

