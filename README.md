# DeepLearning-CIFAR10-Classifier

## Project Overview
This comprehensive project implements a state-of-the-art deep learning-based image classifier for the CIFAR-10 dataset. The implementation includes multiple neural network architectures, extensive preprocessing techniques, and thorough evaluation methodologies. The CIFAR-10 dataset, a benchmark in computer vision, consists of 60,000 32x32 color images across 10 distinct classes, with 6,000 images per class. The dataset provides a balanced representation of common objects:

- **Airplanes**: Commercial and military aircraft in various poses and backgrounds
- **Automobiles**: Various car models including sedans, SUVs, and sports cars
- **Birds**: Different bird species in various natural habitats
- **Cats**: Domestic cats in various poses and environments
- **Deer**: Wild deer in natural settings and various poses
- **Dogs**: Various dog breeds in different environments
- **Frogs**: Amphibians shown in natural settings
- **Horses**: Equines in various poses and environments
- **Ships**: Maritime vessels including sailboats, speedboats, and large ships
- **Trucks**: Commercial and personal trucks and pickups

## Dataset Details
The CIFAR-10 dataset is meticulously curated for machine learning research:

- **Training Data**: 50,000 images
  - **Training Split**: 40,000 images (80% of original training data)
  - **Validation Split**: 10,000 images (20% of original training data)
  - **Split Methodology**: Stratified sampling to maintain class distribution
- **Test Data**: 10,000 images (separate from training/validation)
- **Image Format**: 32x32 RGB color images (3 channels) stored as tensors
- **Class Distribution**: Perfectly balanced with exactly 5,000 training images per class and 1,000 test images per class
- **Total Size**: ~170MB compressed

## Comprehensive Data Preprocessing Pipeline

### Data Cleaning and Validation
- **Duplicate Detection**: Implemented sampling-based approach to check for duplicate images, finding none in the dataset
- **Missing Value Analysis**: Comprehensive check confirmed no missing pixel values or labels
- **Data Consistency Verification**: Ensured all images maintain standard 32x32x3 dimensions with expected pixel ranges

### Data Normalization and Transformation
- **Pixel Normalization**: Scaled all pixel values from integer range [0-255] to floating-point range [0-1] to stabilize gradient descent
- **Categorical Encoding**: Transformed integer class labels [0-9] into one-hot encoded vectors of dimension 10
- **Data Type Conversion**: Converted uint8 image data to float32 for efficient GPU processing

### Data Augmentation Implementation
Implemented robust augmentation techniques to artificially expand the training dataset and improve model generalization:
- **Rotation**: Random rotations within ±15 degrees
- **Width/Height Shifting**: Random shifts up to ±10% along both axes
- **Horizontal Flipping**: Random horizontal flips with 50% probability
- **Zoom Range**: Random zoom between 90-110% of original size
- **Augmentation Pipeline**: Keras ImageDataGenerator with specific parameters for online augmentation during training
  ```python
  datagen = ImageDataGenerator(
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
      zoom_range=0.1
  )
  ```

## Detailed Model Architectures

### 1. Multi-Layer Perceptron (MLP)
A densely connected neural network with progressive dimensionality reduction:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
flatten (Flatten)           (None, 3072)              0         
_________________________________________________________________
dense_1 (Dense)             (None, 1024)              3,146,752 
_________________________________________________________________
dropout (Dropout)           (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)             (None, 512)               524,800   
_________________________________________________________________
dropout_1 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)             (None, 256)               131,328   
_________________________________________________________________
dropout_2 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)             (None, 128)               32,896    
_________________________________________________________________
dropout_3 (Dropout)         (None, 128)               0         
_________________________________________________________________
output (Dense)              (None, 10)                1,290     
=================================================================
Total params: 3,837,066
Trainable params: 3,837,066
Non-trainable params: 0
```

**Architectural Details**:
- **Input Layer**: Flattened 3072-dimensional vector (32×32×3)
- **Hidden Layers**: Four dense layers with decreasing neuron counts (1024→512→256→128)
- **Regularization**: Dropout layers after each hidden layer (0.25 dropout rate)
- **Activation Functions**: ReLU for all hidden layers
- **Output Layer**: Dense layer with 10 neurons and softmax activation
- **Optimization**: Adam optimizer with learning rate of 0.001
- **Loss Function**: Categorical cross-entropy

### 2. Convolutional Neural Network (CNN)
A specialized architecture leveraging spatial relationships in image data:

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 30, 30, 32)        896       
_________________________________________________________________
batch_normalization (BatchN (None, 30, 30, 32)        128       
_________________________________________________________________
conv2d_1 (Conv2D)           (None, 28, 28, 32)        9,248     
_________________________________________________________________
batch_normalization_1 (Batc (None, 28, 28, 32)        128       
_________________________________________________________________
max_pooling2d (MaxPooling2D (None, 14, 14, 32)        0         
_________________________________________________________________
dropout_4 (Dropout)         (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)           (None, 12, 12, 64)        18,496    
_________________________________________________________________
batch_normalization_2 (Batc (None, 12, 12, 64)        256       
_________________________________________________________________
conv2d_3 (Conv2D)           (None, 10, 10, 64)        36,928    
_________________________________________________________________
batch_normalization_3 (Batc (None, 10, 10, 64)        256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling (None, 5, 5, 64)          0         
_________________________________________________________________
dropout_5 (Dropout)         (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_4 (Conv2D)           (None, 3, 3, 128)         73,856    
_________________________________________________________________
batch_normalization_4 (Batc (None, 3, 3, 128)         512       
_________________________________________________________________
flatten_1 (Flatten)         (None, 1152)              0         
_________________________________________________________________
dense_5 (Dense)             (None, 256)               295,168   
_________________________________________________________________
batch_normalization_5 (Batc (None, 256)               1,024     
_________________________________________________________________
dropout_6 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_6 (Dense)             (None, 10)                2,570     
=================================================================
Total params: 439,466
Trainable params: 438,314
Non-trainable params: 1,152
```

**Architectural Details**:
- **Convolutional Blocks**: Three blocks, each with increasing filter counts (32→64→128)
- **Each Block Contains**:
  - Two convolutional layers with 3×3 kernels
  - Batch normalization after each conv layer
  - Max pooling (2×2) at the end of each block
  - Dropout (0.3 rate) for regularization
- **Dense Layers**: One hidden dense layer (256 neurons) after feature extraction
- **Batch Normalization**: Applied after each convolutional and dense layer
- **Activation Functions**: ReLU for all convolutional and dense layers
- **Output Layer**: Dense layer with 10 neurons and softmax activation
- **Optimization**: Adam optimizer with learning rate scheduling
- **Weight Initialization**: He normal initialization for convolutional layers

### 3. Transfer Learning Implementation
Leveraged pre-trained models with customized top layers:

**Base Model**: MobileNetV2 (pre-trained on ImageNet)
```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model

inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.layers.experimental.preprocessing.Resizing(96, 96)(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
```

**Fine-Tuning Strategy**:
1. Train with frozen base model (15 epochs)
2. Unfreeze final convolutional block (5 trainable layers)
3. Fine-tune with lower learning rate (10 epochs)

## Training Methodology
- **Batch Size**: 64 for MLP and CNN, 32 for transfer learning models
- **Training Duration**: 
  - MLP: 50 epochs
  - CNN: 100 epochs with early stopping
  - Transfer Learning: 15 + 10 epochs (two-phase training)
- **Learning Rate Strategy**: 
  - Initial LR: 0.001
  - Schedule: ReduceLROnPlateau with factor=0.1, patience=5
- **Early Stopping**: Patience of 15 epochs monitoring validation loss
- **Hardware Acceleration**: NVIDIA GPU with CUDA acceleration
- **Training Time**:
  - MLP: ~10 minutes
  - CNN: ~45 minutes
  - Transfer Learning: ~60 minutes

## Comprehensive Evaluation Metrics

### Model Performance Comparison
| Model | Test Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|---------------|-----------|--------|----------|---------------|
| MLP   | 55.8%         | 0.56      | 0.56   | 0.56     | 10 min        |
| CNN   | 83.2%         | 0.83      | 0.83   | 0.83     | 45 min        |
| Transfer Learning | 91.7% | 0.92    | 0.92   | 0.92     | 60 min        |

### Per-Class Performance Analysis (CNN Model)
| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Airplane    | 0.86      | 0.82   | 0.84     | 1000    |
| Automobile  | 0.93      | 0.94   | 0.93     | 1000    |
| Bird        | 0.76      | 0.75   | 0.75     | 1000    |
| Cat         | 0.72      | 0.67   | 0.69     | 1000    |
| Deer        | 0.82      | 0.77   | 0.79     | 1000    |
| Dog         | 0.74      | 0.78   | 0.76     | 1000    |
| Frog        | 0.86      | 0.90   | 0.88     | 1000    |
| Horse       | 0.84      | 0.88   | 0.86     | 1000    |
| Ship        | 0.90      | 0.92   | 0.91     | 1000    |
| Truck       | 0.89      | 0.91   | 0.90     | 1000    |

### Error Analysis
- **Confusion Matrix Findings**: Most common misclassifications occurred between:
  - Cat/Dog (9.2% of cat images classified as dogs)
  - Bird/Airplane (7.4% of bird images classified as airplanes)
  - Deer/Horse (5.1% of deer images classified as horses)
- **Difficult Classes**: Categories with lowest accuracy were cats (67%) and birds (75%)
- **Feature Visualization**: Applied t-SNE to visualize the final layer embeddings, revealing clear clustering by class with some overlap in problematic categories

## Project Team (Group 111)
- **Saurabh Jalendra**: Model architecture design, CNN implementation
- **Tushar Shandilya**: Data preprocessing, augmentation pipeline, evaluation metrics
- **Monica Malik**: Transfer learning implementation, hyperparameter optimization
- **Reddy Balaji C**: MLP implementation, error analysis, documentation

## Technologies and Libraries
- **Deep Learning Framework**: TensorFlow 2.18.0, Keras 3.9.0
- **Data Processing**: NumPy 1.24.3, pandas 2.0.3
- **Visualization**: Matplotlib 3.7.2, Seaborn 0.12.2
- **Evaluation**: Scikit-learn 1.3.0
- **Hardware**: NVIDIA GPU with CUDA acceleration
- **Environment**: Python 3.11.7, Anaconda environment

## Future Improvements
- **Architecture Enhancements**: Implement attention mechanisms for better feature selection
- **Ensemble Methods**: Combine predictions from multiple models for improved accuracy
- **Advanced Regularization**: Experiment with techniques like Mixup and CutMix
- **Semi-Supervised Learning**: Leverage unlabeled data to improve generalization
- **Hyperparameter Optimization**: Apply Bayesian optimization for parameter tuning

## References
1. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. 
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
3. Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions.
4. Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., & Le, Q. V. (2019). AutoAugment: Learning Augmentation Policies from Data.