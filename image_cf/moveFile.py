import os
import shutil


def mvFile(type):
    rgbFilePath = 'I:\\cfimage\\rgb\\'+type+'\\'
    srcFilePath = 'I:\\facade\\facade\\train_label\\'
    sarFilePath = 'I:\\cfimage\\sar\\'+type+'\\'
    fileList = os.listdir(rgbFilePath)
    for file in fileList:
        sarFile = file.replace('s2', 's1')
        srcPath = srcFilePath + sarFile
        dstPath = sarFilePath + sarFile
        shutil.move(srcPath, dstPath)


mvFile('beach')
# mvFile('field')
mvFile('forest')
mvFile('mountain')
mvFile('sea')