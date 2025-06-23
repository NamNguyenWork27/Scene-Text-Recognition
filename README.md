# Introduction 

Scene Text Recognition is a problem that applies image processing and handwriting recognition techniques to identify texts appearing in photos taken from real-world environments. This problem has many practical applications, such as:
- Text processing in images: Recognizing text in documents, newspapers, signs, etc.
- Information retrieval: Recognizing text in images on the internet to collect important data.
- Process automation: Recognizing text in images to automate tasks, such as processing
orders, payments, etc.

A typical Scene Text Recognition process consists of two main stages:
- Detector: Locate text blocks in images.
- Recognizer: Decode text at identified locations.

![Pipeline](Images\two_stage_approach.png)

In this project, I will develop a Text Recognition in Images program using YOLOv11 (for text detection) and CRNN (for character recognition). The input and output of the program are as follows:
- Input: An image containing text.
- Output: Location coordinates and text content in the image.


# Setup Program

1. Data Loading: The modules in the Text in Image Recognition program of this project, including Text Detection and Text Recognition, will be trained on the ICDAR2003 dataset.

2. Setting up the Text Detection module: In this module, we will build a program that receives an input image and returns the coordinates surrounding all the text in the image. Specifically, we will use YOLOv11 to do this job. Input/output of this problem in the following figure:

    ![Text_Detection](Images\Input_Output_Detection.png)

3. Install Text Recognition module: In this module, we will build a program
that receives an image containing only text and returns the text content as a string. Input and output (I/O) of this problem in more detail in the image below:

    ![Text_Recognition](Images\Input_Output_Text_Recognition.png)

    To solve this problem, we will choose the Convolutional Recurrent Neural Networks (CRNN) model, a basic but effective model. This is one of the first models developed to solve data problems that are both image and sequence. The structure of the CRNN model is as follows:

    ![CRNN_Model](Images\CRNN.png)

    This model combines CNN (Convolutional Neural Networks) and RNN (Recurrent Neural Networks) to extract features from images and text strings in images. At the same time, CRNN also uses a special loss function, CTC Loss, to improve the model's results.


4. Install the entire pipeline: After completing the construction of the two Text Detection and Text Recognition models, we will build the complete Scene Text Recognition program by combining the two models above.

# Results 
Some results of the model: 

![Results](Images\Results.png)

---

# Installation
To run this project, install the required dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```

# Usage 

1. Clone the repository: 

    ```bash
    git https://github.com/NamNguyenWork27/Scene-Text-Recognition.git
    ```


2. Run deploy local:

    Before:

    ```bash
    uvicorn Deployment.main:app --reload
    ```

    After: 

    ```bash
    streamlit run Deployment/app.py
    ```

    Make sure your device is equipped with GPU otherwise run load state dict with command: reg_model.load_state_dict(torch.load("ocr_crnn.pt", map_location=torch.device("cpu")))

# Test on local deployment

**Before**:
![Original](Images/Original.png)

**After**
![After_runOCR](Images/After_runOCR.png)
