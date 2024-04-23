from ultralytics import YOLO

def main():
    model = YOLO(r"Divide-10-im\i\weights\best.pt")
    model.val(data=r"C:\data\Divide-10\i\data.yaml", batch=1,split='val')

if __name__ == '__main__':
    main()
