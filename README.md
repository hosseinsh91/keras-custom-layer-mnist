# **MNIST Custom Layer Classification**

## **📌 Project Overview**
This project implements a **deep learning model** for **MNIST digit classification** using **TensorFlow/Keras**. A **custom Keras layer** is designed to manipulate input data using **non-linear transformations** before passing it to fully connected layers.

### **🚀 Key Features**
✅ **Preprocessing the MNIST dataset**  
✅ **Creating a custom Keras layer with trainable weights**  
✅ **Training a deep learning model with ReLU activation**  
✅ **Comparing performance against a standard dense model**  
✅ **Plotting accuracy and validation loss comparisons**  

---

## **📌 Dataset: MNIST**
The **MNIST dataset** consists of **60,000 training images** and **10,000 test images**, each containing handwritten digits from 0-9.

### **📌 Data Preprocessing**
A subset of digits (0, 2, 5) is selected for classification:
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

selected_labels = [0, 2, 5]
mask = np.isin(y_train, selected_labels)
x_train_f = x_train[mask]
y_train_f = y_train[mask]

# Mapping selected labels to 0, 1, 2 for classification
label_map = {0: 0, 2: 1, 5: 2}
y_train_f = np.array([label_map[label] for label in y_train_f])

# Normalize pixel values
x_train_f = x_train_f / 255.0
```

---

## **📌 Custom Keras Layer**
A **custom Keras layer** is defined to **apply a transformation on input data** before passing it to the next layer.
```python
class MyLayer(keras.layers.Layer):
    def __init__(self, units):
        super(MyLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.v = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="random_normal",
                                 trainable=True)
    
    def call(self, inputs):
        return tf.matmul(tf.pow(inputs, 3), self.v) - tf.matmul(inputs, self.w) + self.b
```

---

## **📌 Model Architecture**
Two models are implemented:
### **1️⃣ Custom Layer Model**
```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(150, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    MyLayer(units=50),
    keras.layers.Dense(3, activation='softmax')
])
```

### **2️⃣ Simple Dense Model**
```python
model2 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
```

---

## **📌 Model Compilation & Training**
Both models are compiled using **SGD optimizer** and trained on **50 epochs**.

### **Train Custom Layer Model**
```python
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train_f, y_train_f, epochs=50, validation_split=0.2)
```

### **Train Simple Dense Model**
```python
model2.compile(optimizer='sgd',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history2 = model2.fit(x_train_f, y_train_f, epochs=50, validation_split=0.2)
```

---

## **📌 Model Performance Comparison**
A comparison between the **custom layer model** and **simple dense model** is plotted.
```python
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(history.history["val_loss"], label="val_loss_Custom_Layer")
ax.plot(history2.history["val_loss"], label="val_loss_Simple_Layer")
ax.plot(history.history["val_accuracy"], label="val_accuracy_Custom_Layer")
ax.plot(history2.history["val_accuracy"], label="val_accuracy_Simple_Layer")
ax.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.title("Comparison of Custom Layer vs. Simple Model")
plt.show()
```

### **📊 Key Observations**
✅ **Custom Layer Model achieved better accuracy over time**  
✅ **Validation loss decreased significantly for the custom model**  
✅ **Non-linear transformation in MyLayer improved feature extraction**  

---

## **📌 Installation & Setup**
### **📌 Prerequisites**
- **Python 3.x**
- **Jupyter Notebook**
- **TensorFlow, NumPy, Matplotlib**

### **📌 Install Required Libraries**
```bash
pip install tensorflow numpy matplotlib
```

---

## **📌 Running the Notebook**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YourGitHubUsername/mnist-custom-layer-classification.git
cd mnist-custom-layer-classification
```

### **2️⃣ Launch Jupyter Notebook**
```bash
jupyter notebook
```

### **3️⃣ Run the Notebook**
Open `mnist_custom_layer.ipynb` and execute all cells.

---

## **📌 Conclusion**
This project demonstrates how **custom layers can enhance deep learning models** by applying transformations beyond standard activations. The comparison between **a traditional dense model and a custom transformation layer** showcases **the impact of feature manipulation on model performance**.

---

## **📌 License**
This project is licensed under the **MIT License**.

