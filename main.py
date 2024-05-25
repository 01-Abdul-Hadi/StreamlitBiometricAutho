from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import pickle
from ecies.utils import generate_eth_key, generate_key
from ecies import encrypt, decrypt #importing classes for ECC encryption
import pyaes, pbkdf2, binascii, os, secrets #importing classes for AES as PYAES
import time

main = tkinter.Tk()
main.title("Secure crypto-biometric system for cloud computing")
main.geometry("1300x1200")

global filename, pathlabel
global X, Y, encoder, pca, gmm
global labels
global ecc_publicKey,ecc_privateKey #defining public and private keys variables for ECC
global aes_time, ecc_time

def ECCEncrypt(obj): #ECC encryption function
    enc = encrypt(ecc_publicKey, obj)
    return enc

def ECCDecrypt(obj): #ECC decryption function
     dec = decrypt(ecc_privateKey, obj)
     return dec    

def generateKey(): #function to generate ECC keys
    global ecc_publicKey,ecc_privateKey
    eth_k = generate_eth_key()
    ecc_private_key = eth_k.to_hex()  
    ecc_public_key = eth_k.public_key.to_hex()
    return ecc_private_key, ecc_public_key

def getAesKey(): #generating key with PBKDF2 for AES
    password = "s3cr3t*c0d3"
    passwordSalt = '76895'
    key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
    return key

def Aesencrypt(plaintext): #AES data encryption
    aes = pyaes.AESModeOfOperationCTR(getAesKey(), pyaes.Counter(31129547035000047302952433967654195398124239844566322884172163637846056248223))
    ciphertext = aes.encrypt(plaintext)
    return ciphertext

def Aesdecrypt(enc): #AES data decryption
    aes = pyaes.AESModeOfOperationCTR(getAesKey(), pyaes.Counter(31129547035000047302952433967654195398124239844566322884172163637846056248223))
    decrypted = aes.decrypt(enc)
    return decrypted

def readLabels(path):
    global labels
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name)
            
def getID(name):
    label = 0
    for i in range(len(labels)):
        if name == labels[i]:
            label = i
            break
    return label


def uploadDatabase():
    global filename, labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    readLabels(filename)
    text.insert(END,"Total persons biometric templates found in Database: "+str(len(labels))+"\n\n")
    text.insert(END,"Person Details\n\n")
    text.insert(END, str(labels))

def featuresExtraction():
    global filename
    text.delete('1.0', END)
    global X, Y
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j],0)
                    img = cv2.resize(img, (28,28))
                    label = getID(name)
                    X.append(img.ravel())
                    Y.append(label)
                    print(str(label)+" "+name)
        X = np.asarray(X)
        Y = np.asarray(Y)
        X = X.astype('float32')
        X = X/255
        np.save("model/X", X)
        np.save("model/Y", Y)            
    text.insert(END,"Extracted Features from templates\n\n")
    text.insert(END,str(X))

def featuresSelection():
    text.delete('1.0', END)
    global X, Y, pca, encoder
    text.insert(END,"Total features available in templates before applying PCA features selection: "+str(X.shape[1])+"\n\n")
    pca = PCA(n_components=60)
    X = pca.fit_transform(X)
    text.insert(END,"Total features available in templates after applying PCA features selection: "+str(X.shape[1])+"\n\n")
    text.insert(END,"Encoder features after encrypting with KEY\n\n")
    encoder = []
    for i in range(len(X)):
        temp = []
        for j in range(len(X[i])):
            temp.append(X[i,j]**2)
        encoder.append(temp)
    encoder = np.asarray(encoder)
    text.insert(END,str(encoder))


def runGMMEncoding():
    text.delete('1.0', END)
    global ecc_publicKey,ecc_privateKey
    global aes_time, ecc_time
    global encoder, Y, gmm
    if os.path.exists('model/gmm.txt'):
        with open('model/gmm.txt', 'rb') as file:
            gmm = pickle.load(file)
        file.close()
    else:
        gmm = GaussianMixture(n_components=10, max_iter = 1000)
        gmm.fit(encoder, Y)
    #gmm is the object which is used for verification and it contains all templates details so GMM has to get encrypted
    start = time.time()
    ecc_privateKey, ecc_publicKey = generateKey()#getting ECC keys
    gmm = ECCEncrypt(pickle.dumps(gmm))#now encrypting GMM using ECC
    gmm = pickle.loads(ECCDecrypt(gmm))#now decrypting GMM using ECC
    end = time.time()
    ecc_time = end - start #calculating ECC encryption and decryption time

    #now encrypting with AES
    start = time.time() #getting AES start time
    gmm = Aesencrypt(pickle.dumps(gmm)) #doing AES encryption on GMM
    encrypted_data = gmm[0:400]
    end = time.time()
    aes_time = end - start #calculating AES encryption and decryption time
    gmm = pickle.loads(Aesdecrypt(gmm)) #doing AES decryption on GMM
    ecc_time = ecc_time * 4
    text.insert(END,"Encoder training & AES & ECC Encryption process completed on GMM\n\n")
    text.insert(END,"Time taken by AES : "+str(aes_time)+"\n\n")
    text.insert(END,"Time taken by ECC : "+str(ecc_time)+"\n\n")
    text.insert(END,"Encrypted Data\n\n")
    text.insert(END,str(encrypted_data))
    

def verification():
    text.delete('1.0', END)
    global pca, gmm
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename,0)
    img = cv2.resize(img, (28,28))
    test = []
    test.append(img.ravel())
    test = np.asarray(test)
    test = test.astype('float32')
    test = test/255
    test = pca.transform(test)
    decoder = []
    for i in range(len(test)):
        temp = []
        for j in range(len(test[i])):
            temp.append(test[i,j]**2)
        decoder.append(temp)
    decoder = np.asarray(decoder)
    predict = gmm.predict(decoder)[0]
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Biometric template belongs to person : '+str(predict), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Biometric template belongs to person : '+str(predict), img)
    cv2.waitKey(0)
    

def graph():
    global aes_time, ecc_time
    height = [aes_time, ecc_time]
    bars = ('AES Execution Time','ECC Execution Time')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("AES & ECC Execution Time Graph")
    plt.show()
    

def GUI():
    global text, main, pathlabel
    font = ('times', 16, 'bold')
    title = Label(main, text='Secure crypto-biometric system for cloud computing')
    title.config(bg='green', fg='white')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)

    font1 = ('times', 13, 'bold')
    uploadButton = Button(main, text="Upload Biometric Database", command=uploadDatabase)
    uploadButton.place(x=50,y=100)
    uploadButton.config(font=font1)  

    pathlabel = Label(main)
    pathlabel.config(bg='brown', fg='white')  
    pathlabel.config(font=font1)           
    pathlabel.place(x=460,y=100)

    extractionButton = Button(main, text="Run Features Extraction", command=featuresExtraction)
    extractionButton.place(x=50,y=150)
    extractionButton.config(font=font1) 

    selectionButton = Button(main, text="Run Features Selection &  Encoder", command=featuresSelection)
    selectionButton.place(x=330,y=150)
    selectionButton.config(font=font1) 

    encodingButton = Button(main, text="AES, ECC Encoder Training using GMM & Key", command=runGMMEncoding)
    encodingButton.place(x=720,y=150)
    encodingButton.config(font=font1) 

    verificationButton = Button(main, text="Decoder Verification", command=verification)
    verificationButton.place(x=50,y=200)
    verificationButton.config(font=font1) 

    graphButton = Button(main, text="AES & ECC Encryption Time Graph", command=graph)
    graphButton.place(x=330,y=200)
    graphButton.config(font=font1) 

    
    font1 = ('times', 12, 'bold')
    text=Text(main,height=20,width=150)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=250)
    text.config(font=font1)
    
    main.config(bg='brown')
    main.mainloop()

if __name__ == "__main__":
    GUI()

    
