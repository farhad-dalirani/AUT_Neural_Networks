import threading

tensorBoardPath = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder10\\on\\'

def launchTensorBoard():
    import os
    print(os.system('cd'))
    #os.system('tensorboard --logdir=' + tensorBoardPath)
    os.system('tensorboard --logdir '+tensorBoardPath)
    return

t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()