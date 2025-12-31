import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing import image

# -------------------------- 1. 配置参数 --------------------------
# 数据集路径（请替换为你解压后的数据集路径）
DATASET_PATH = "/path/to/your/dog-and-cat-classification-dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# 图像参数
IMG_WIDTH, IMG_HEIGHT = 150, 150  # 统一图像尺寸
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2  # 猫/狗二分类

# -------------------------- 2. 数据预处理与增强 --------------------------
# 训练集数据增强（防止过拟合）
train_datagen = ImageDataGenerator(
    rescale=1./255,               # 归一化到0-1
    rotation_range=40,            # 随机旋转±40度
    width_shift_range=0.2,        # 随机水平平移
    height_shift_range=0.2,       # 随机垂直平移
    shear_range=0.2,              # 随机剪切
    zoom_range=0.2,               # 随机缩放
    horizontal_flip=True,         # 随机水平翻转
    fill_mode='nearest'           # 填充方式
)

# 测试集仅归一化（不增强）
test_datagen = ImageDataGenerator(rescale=1./255)

# 加载训练集
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # 二分类用binary，多分类用categorical
)

# 加载测试集
test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # 评估时保持顺序
)

# -------------------------- 3. 构建CNN模型 --------------------------
model = Sequential([
    # 第一层卷积
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    BatchNormalization(),  # 批量归一化加速训练
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),  # Dropout防止过拟合

    # 第二层卷积
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 第三层卷积
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 全连接层
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 二分类输出层（sigmoid输出0-1）
])

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # 二分类损失函数
    metrics=['accuracy']
)

# 打印模型结构
model.summary()

# -------------------------- 4. 训练模型 --------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1  # 显示训练过程
)

# -------------------------- 5. 评估模型 --------------------------
# 在测试集上评估
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\n测试集准确率: {test_acc:.4f}, 测试集损失: {test_loss:.4f}")

# -------------------------- 6. 可视化训练过程 --------------------------
def plot_training_history(history):
    # 准确率曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# 绘制曲线
plot_training_history(history)

# -------------------------- 7. 单张图片预测函数 --------------------------
def predict_image(img_path):
    # 加载并预处理图片
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 增加batch维度
    img_array /= 255.0  # 归一化

    # 预测
    prediction = model.predict(img_array)[0][0]
    class_names = {0: '猫', 1: '狗'}  # 对应generator的class_indices
    result = class_names[0] if prediction < 0.5 else class_names[1]
    confidence = 1 - prediction if prediction < 0.5 else prediction

    # 显示图片和结果
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"预测结果: {result} (置信度: {confidence:.4f})")
    plt.show()

    return result, confidence

# -------------------------- 8. 测试单张图片 --------------------------
# 替换为你的测试图片路径
TEST_IMG_PATH = "/path/to/your/test_image.jpg"
predict_image(TEST_IMG_PATH)

# -------------------------- 9. 保存模型 --------------------------
model.save("cat_dog_classifier.h5")
print("模型已保存为 cat_dog_classifier.h5")
