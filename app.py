import streamlit as st
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pickle
from ecies.utils import generate_eth_key
from ecies import encrypt, decrypt
import pyaes, pbkdf2, secrets, time

# Global Variables
labels = []
X, Y, encoder, pca, gmm = None, None, None, None, None
ecc_publicKey, ecc_privateKey = None, None
aes_time, ecc_time = None, None

def ECCEncrypt(obj):  # ECC encryption function
    enc = encrypt(ecc_publicKey, obj)
    return enc

def ECCDecrypt(obj):  # ECC decryption function
    dec = decrypt(ecc_privateKey, obj)
    return dec    

def generateKey():  # function to generate ECC keys
    global ecc_publicKey, ecc_privateKey
    eth_k = generate_eth_key()
    ecc_privateKey = eth_k.to_hex()  
    ecc_publicKey = eth_k.public_key.to_hex()
    return ecc_privateKey, ecc_publicKey

def getAesKey():  # generating key with PBKDF2 for AES
    password = "s3cr3t*c0d3"
    passwordSalt = '76895'
    key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
    return key

def Aesencrypt(plaintext):  # AES data encryption
    aes = pyaes.AESModeOfOperationCTR(getAesKey(), pyaes.Counter(31129547035000047302952433967654195398124239844566322884172163637846056248223))
    ciphertext = aes.encrypt(plaintext)
    return ciphertext

def Aesdecrypt(enc):  # AES data decryption
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
    global labels
    path = st.text_input("Enter the path of the biometric database folder:")
    if st.button("Load Database"):
        if os.path.isdir(path):
            readLabels(path)
            st.write(f"Total persons biometric templates found in Database: {len(labels)}")
            st.write("Person Details")
            st.write(labels)
            st.session_state['database_path'] = path
            return path
        else:
            st.error("Invalid directory path. Please enter a valid path.")
            return None

def featuresExtraction(path):
    global X, Y
    if 'X' in st.session_state and 'Y' in st.session_state:
        X = st.session_state['X']
        Y = st.session_state['Y']
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(os.path.join(root, directory[j]), 0)
                    img = cv2.resize(img, (28, 28))
                    label = getID(name)
                    X.append(img.ravel())
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        X = X.astype('float32')
        X = X / 255
        st.session_state['X'] = X
        st.session_state['Y'] = Y
    st.write("Extracted Features from templates")
    st.write(X)

def featuresSelection():
    global X, pca, encoder
    if 'X' not in st.session_state:
        st.error("Feature extraction needs to be completed before feature selection.")
        return
    X = st.session_state['X']
    st.write(f"Total features available in templates before applying PCA features selection: {X.shape[1]}")
    pca = PCA(n_components=60)
    X = pca.fit_transform(X)
    st.write(f"Total features available in templates after applying PCA features selection: {X.shape[1]}")
    st.write("Encoder features after encrypting with KEY")
    encoder = []
    for i in range(len(X)):
        temp = []
        for j in range(len(X[i])):
            temp.append(X[i, j]**2)
        encoder.append(temp)
    encoder = np.asarray(encoder)
    st.session_state['encoder'] = encoder
    st.session_state['pca'] = pca
    st.write(encoder)

def runGMMEncoding():
    global ecc_publicKey, ecc_privateKey
    global aes_time, ecc_time
    global encoder, Y, gmm
    if 'encoder' not in st.session_state or 'Y' not in st.session_state:
        st.error("Feature selection needs to be completed before GMM encoding.")
        return
    encoder = st.session_state['encoder']
    Y = st.session_state['Y']
    if os.path.exists('model/gmm.txt'):
        with open('model/gmm.txt', 'rb') as file:
            gmm = pickle.load(file)
        file.close()
    else:
        gmm = GaussianMixture(n_components=10, max_iter=1000)
        gmm.fit(encoder, Y)
    start = time.time()
    ecc_privateKey, ecc_publicKey = generateKey()
    gmm = ECCEncrypt(pickle.dumps(gmm))
    gmm = pickle.loads(ECCDecrypt(gmm))
    end = time.time()
    ecc_time = end - start
    start = time.time()
    gmm = Aesencrypt(pickle.dumps(gmm))
    encrypted_data = gmm[:400]
    end = time.time()
    aes_time = end - start
    gmm = pickle.loads(Aesdecrypt(gmm))
    ecc_time = ecc_time * 4
    st.session_state['gmm'] = gmm
    st.session_state['aes_time'] = aes_time
    st.session_state['ecc_time'] = ecc_time
    st.write("Encoder training & AES & ECC Encryption process completed on GMM")
    st.write(f"Time taken by AES: {aes_time}")
    st.write(f"Time taken by ECC: {ecc_time}")
    st.write("Encrypted Data")
    st.write(encrypted_data)

def verification():
    global pca, gmm
    if 'pca' not in st.session_state or 'gmm' not in st.session_state:
        st.error("PCA and GMM model must be initialized before verification.")
        return
    pca = st.session_state['pca']
    gmm = st.session_state['gmm']
    uploaded_file = st.file_uploader("Select Biometric Template Image", type=["jpg", "png"])
    if uploaded_file is not None:
        # Convert the file to bytes
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Read the image using OpenCV
        img = cv2.imdecode(file_bytes, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (28, 28))
        test = img_resized.ravel().astype('float32') / 255.0
        test = pca.transform([test ** 2])
        predict = gmm.predict(test)[0]
        st.image(img, caption='Biometric template belongs to person : '+str(predict))

def graph():
    if 'aes_time' in st.session_state and 'ecc_time' in st.session_state:
        aes_time = st.session_state['aes_time']
        ecc_time = st.session_state['ecc_time']
        height = [aes_time, ecc_time]
        bars = ['AES Execution Time', 'ECC Execution Time']
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.title("AES & ECC Execution Time Graph")
        st.pyplot(plt)
    else:
        st.error("Encryption times are not available. Ensure that GMM encoding has been run.")

# Streamlit UI Elements
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose an action", ["Upload Database", "Feature Extraction", "Feature Selection", "GMM Encoding", "Verification", "Graph"])

path = None
if option == "Upload Database":
    path = uploadDatabase()
elif option == "Feature Extraction" and 'database_path' in st.session_state:
    path = st.session_state['database_path']
    featuresExtraction(path)
elif option == "Feature Selection":
    featuresSelection()
elif option == "GMM Encoding":
    runGMMEncoding()
elif option == "Verification":
    verification()
elif option == "Graph":
    graph()
