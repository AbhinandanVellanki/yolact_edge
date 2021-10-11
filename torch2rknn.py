#to convert a pytorch model into RKNN
from rknn.api import RKNN

rknn = RKNN()
print(rknn.load_pytorch(model='./yolact_edge_resnet50_54_800000.pth', input_size_list=[[3, 224, 224]], convert_engine="torch1.2"))