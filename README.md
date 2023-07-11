# Official YOLOv7 trained on a custom dataset for window detection Open/Closed

## Testing object detection 

On video:
``` shell
python detect.py --weights yolo.pt --conf 0.25 --img-size 640 --source yourvideo.mp4 
```


On camera:
``` shell
python detect.py --weights yolo.pt --conf 0.25 --img-size 640 --source 0  
```

On image:
``` shell
python detect.py --weights yolo.pt --conf 0.25 --img-size 640 --source inference/images/image.jpg
```

## Testing window detection 

On video:
``` shell
python detect.py --weights window.pt --conf 0.25 --img-size 640 --source yourvideo.mp4 
```


On camera:
``` shell
python detect.py --weights window.pt --conf 0.25 --img-size 640 --source 0  
```

On image:
``` shell
python detect.py --weights window.pt --conf 0.25 --img-size 640 --source inference/images/image.jpg
```

