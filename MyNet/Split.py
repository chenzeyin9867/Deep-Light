import os
import time
import numpy as np
from IPython.display import clear_output
sourceDir="./40000data/skybox/"
sourceMaskDir = "./SkyBoxAddMask40000"
#trainDir validationDir testDir
targetDir=["./traindir32000",
           "./validatedir4000",
           "./testdir4000"]
MaskTargetDir = ["./trainMaskdir32000",
           "./validateMaskdir4000",
           "./testdirMask4000"]

fileList1=os.listdir(sourceDir)
fileList2 = os.listdir(sourceMaskDir)
fileNum=len(fileList1)
fileList1.sort()
fileList2.sort()

x=int(8./10*fileNum)
y=int(1./10*fileNum)
z=fileNum-x-y
targetFileNum=[x,x+y,x+y+z]
print(x,y,z)
#随机数据下标
index=np.array(range(fileNum))
np.random.shuffle(index)
#将数据划分到训练集 验证集和测试集三个文件夹中
targetDirId=0
outputFileNum=0
for i in index:
    f = fileList1[i]
    g = fileList2[i]
    sourceF=os.path.join(sourceDir,f)
    sourceMaskF = os.path.join(sourceMaskDir,g)
    #print("targetDirId: ",targetDirId)
    targetF=os.path.join(targetDir[targetDirId],f)
    targetMaskF = os.path.join(MaskTargetDir[targetDirId],g)
    if os.path.isfile(sourceF):
        if not os.path.exists(targetDir[targetDirId]):
            os.makedirs(targetDir[targetDirId])
        if not os.path.exists(MaskTargetDir[targetDirId]):
            os.makedirs(MaskTargetDir[targetDirId])
        outputFileNum+=1
        open(targetF,"wb").write(open(sourceF,"rb").read())
        open(targetMaskF,"wb").write(open(sourceMaskF,"rb").read())
        if outputFileNum>=targetFileNum[targetDirId]:
            targetDirId+=1
        #clear_output(wait=True)
        #print("{}/{}".format(outputFileNum,fileNum))


# In[46]:
