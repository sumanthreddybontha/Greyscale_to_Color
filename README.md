# Greyscale_to_Color
Computer Vision Project

# Greyscale_to_Color
Computer Vision Project

Image Colorization using Autoencoder and U-Net - ReadMe
=======================================================

This repository demonstrates the use of both **Autoencoder-based deep learning models** and **U-Net architecture** for colorizing grayscale images. The code is implemented using TensorFlow and Keras, and the training process leverages both **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** for loss calculation. The model aims to predict the AB channels of the LAB color space, producing realistic colorizations from grayscale inputs.

Requirements
------------

To run this code, you will need to install the following libraries:

### Required Libraries:

*   **TensorFlow 2.x** (for model building, training, and evaluation)
    
*   **Keras** (part of TensorFlow, used for creating deep learning models)
    
*   **OpenCV** (for image loading and preprocessing)
    
*   **scikit-image** (for color space conversion: RGB to LAB)
    
*   **NumPy** (for numerical computations)
    
*   **Matplotlib** (for visualization)
    
*   **KaggleHub** (for efficient dataset management and usage)
    

### Installation:

You can install all required libraries using the following command:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopy codepip install tensorflow opencv-python scikit-image numpy matplotlib tqdm kagglehub   `

For GPU support, install TensorFlow GPU:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopy codepip install tensorflow-gpu   `

Ensure you have the appropriate CUDA and cuDNN versions installed to leverage GPU acceleration. Check TensorFlow's official installation guide for details.

Model Overview
--------------

### 1\. **Autoencoder-based Colorization** (Convolution Auto-Encoder Cv.ipynb)

This part of the project uses a **convolutional autoencoder** architecture to colorize grayscale images. The encoder compresses the input image, while the decoder reconstructs the output colorized image. The model is trained using the **MSE loss function**, which helps to minimize the pixel-wise error between the predicted and ground truth AB channels.

*   **Loss Functions**: MSE and MAE (for comparison)
    
*   **Optimizer**: Adam optimizer
    
*   **Input**: Grayscale image (1 channel)
    
*   **Output**: Predicted AB channels (2 channels in LAB color space)
    

### 2\. **U-Net Architecture** (AutoEncoder U-Net.ipynb)

This model uses the **U-Net architecture**, which is popular for image segmentation and colorization tasks. U-Net uses an encoder-decoder structure with skip connections that help preserve spatial features during upsampling.

*   **Encoder**: Extracts hierarchical features using convolutional layers.
    
*   **Decoder**: Upsamples the features to generate the final colorized image.
    
*   **Loss Function**: MSE for better colorization quality.
    

Both models are trained on grayscale images, and the network learns to predict color information in the AB channels of the LAB color space.

How to Run - Convolution AutoEncoder
------------------------------------

1.  **Prepare Your Data**: Ensure you have the dataset by running first few cells as it connects to Kaggle.
    
2.  **Train the Model**: Run all the cells sequentially to train either the autoencoder model or the U-Net model based on your preference.
    
3.  **Generate Colorized Images**: After training, use the models to colorize new grayscale images.
    

How to Run - U-Net AutoEncoder
------------------------------

1.  **Prepare Your Data**: Upload the dataset ([https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization)) to the runtime in the current directory.
    
2.  **Train the Model**: Run all the cells sequentially to train either the autoencoder model or the U-Net model based on your preference.
    
3.  **Generate Colorized Images**: After training, use the models to colorize new grayscale images.

