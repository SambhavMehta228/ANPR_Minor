import os
import ultralytics
from ultralytics import YOLO
import multiprocessing
from multiprocessing import Process, freeze_support
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train():
   model=YOLO("C:\\Users\\Sambhav Mehta\\Desktop\\ANPR_Minor\\last.pt")
   results=model.train(data="C:\\Users\\Sambhav Mehta\\Desktop\\ANPR_Minor\\abc.yaml", epochs=60) 

if __name__ == '__main__':  
  freeze_support()
  Process(target=train).start()
                    