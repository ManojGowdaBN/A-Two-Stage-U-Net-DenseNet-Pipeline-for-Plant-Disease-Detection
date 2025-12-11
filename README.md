# A-Two-Stage-U-Net-DenseNet-Pipeline-for-Plant-Disease-Detection
Project Overview
This project implements a two-stage deep learning pipeline to detect plant leaf diseases. In Stage 1, a U-Net model segments the leaf regions from raw images, and in Stage 2, a DenseNet121 classifier predicts the disease class from the segmented leaf. This approach helps isolate the relevant plant features and ignore complex backgrounds, improving accuracy. By focusing on pixel-level leaf regions, the segmentation model filters out clutter (such as soil, stems or debris) so that the classifier can concentrate on disease symptoms. In practice, the two-stage pipeline (segmentation → classification) has been shown to remove irrelevant context and increase specificity, leading to better performance on challenging field images. We classify leaves into 8 categories: Bacterial_Spot, Black_Measles, Black_Rot, Gray_Leaf_Spot, Healthy, Powdery_mildew, Rust, and Scab.

# Folder Structure
The repository uses a clear directory layout for data and models, for example:
A-Two-Stage-U-Net-DenseNet-Pipeline-for-Plant-Disease-Detection/
├── data/  
│   ├── Train/             # training images (and masks) for segmentation  
│   ├── Validation/        # validation split for segmentation  
│   ├── Test/              # held-out test images for segmentation  
│   └── segmented/         # output folder for leaf-only images (after Stage 1)  
├── models/                # saved model files (U-Net and DenseNet weights)  
├── notebooks/             # Jupyter notebooks for experimentation (if any)  
└── README.md              # this documentation file  

Train, Validation, Test: Each of these contains the original images and corresponding ground-truth masks for that split. The U-Net is trained on the “Train” set and evaluated on the “Validation” split during training. The “Test” set is reserved for final evaluation.

segmented: After Stage 1 training, this folder holds the segmented leaf images (or masks applied to images). These segmented images are used as inputs to the classification stage (Stage 2). In other words, the U-Net’s output (leaf regions) are saved here for later use by the DenseNet classifier.m

# Dataset

The dataset consists of leaf images labeled by disease class. For segmentation, each image has an associated binary mask indicating leaf vs. background. Masks are generated automatically via an HSV color filter: we convert images to the HSV color space and threshold on hue/saturation values to isolate green leaf regions from the background. Both the original images and mask images are resized to 256×256 pixels before training the U-Net. This fixed-size preprocessing ensures consistent input dimensions and speeds up training. After segmentation, the resulting leaf-only images are resized or padded as needed to 224×224 for input to the DenseNet classifier (as required by the DenseNet121 architecture).

# Model Architecture

Stage 1 – U-Net Segmentation: We implemented a standard U-Net in TensorFlow/Keras. U-Net is an encoder–decoder CNN with skip connections. The encoder (“contracting”) path applies a series of convolution and pooling layers to capture contextual features, while the decoder (“expanding”) path upsamples via transposed convolutions and concatenates corresponding encoder feature maps via skip connections. This symmetric U-shaped structure enables precise localization of the leaf region. In our code, the U-Net has multiple downsampling blocks (with increasing filters, e.g. 64→128→256→512) and matching upsampling blocks. The output layer is a 1-channel sigmoid (for binary mask) that separates leaf from background.

Stage 2 – DenseNet121 Classification: We use a DenseNet121 model from TensorFlow/Keras Applications (pre-trained on ImageNet). DenseNet121, introduced by Huang et al. (CVPR 2017), connects each layer to all subsequent layers, which alleviates vanishing gradients and encourages feature reuse. We instantiate tf.keras.applications.DenseNet121(include_top=True, weights='imagenet') and replace the final dense layer to output 8 classes (with a softmax activation). The model takes 224×224×3 RGB inputs (segmented leaf images) and outputs probabilities for the 8 disease categories. During fine-tuning, all ImageNet weights serve as initialization, and we train the top layers on our plant dataset.

# Training
Both models were trained using the Adam optimizer with standard settings. Key training details: we used a batch size of 16. For the DenseNet classifier, input images are resized to 224×224 (as required by the architecture). We employed Keras callbacks for robust training: EarlyStopping to halt training when validation loss stops improving, and ReduceLROnPlateau to reduce the learning rate when learning stagnates
. These techniques help prevent overfitting and ensure convergence.

# Results

The two-stage model achieves high accuracy on the plant disease task. As training progresses, the U-Net quickly learns to isolate leaf regions (achieving high mask accuracy), and the DenseNet classifier reaches strong performance on the 8-class problem. In our tests, the DenseNet’s training accuracy rose to ~95%, with validation accuracy around the mid-90s% range, while both training and validation losses decreased smoothly. These trends indicate the model effectively learned to classify diseases from the segmented leaf images. (Actual results and learning curves can be found in the project’s experiment logs.)
