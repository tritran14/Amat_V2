import os
import base64

PARENT_DIR = "train_img/"

def create_dir(userId, base64List):
    path = os.path.join(PARENT_DIR, "{}/".format(userId))
    # print("path : {}".format(path))
    os.mkdir(path)
    for base64Ele in base64List:
        fileNamePattern = "{}_{}.jpg"
        filePath = os.path.join(path, fileNamePattern.format(userId, base64Ele["id"]))
        base64Value = base64Ele["value"]
        with open(filePath, "wb") as fh:
            fh.write(base64.b64decode(base64Value))