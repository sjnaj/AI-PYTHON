import numpy as np
import os
import urllib.request
import gzip
from matplotlib import pyplot as plt
# 下载并解压 MNIST 数据集
def download_mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    if not os.path.exists('mnist'):
        os.makedirs('mnist')

    for filename in filenames:
        file_path = os.path.join('mnist', filename)
        if not os.path.exists(file_path):
            print(f'Downloading {filename}...')
            url = base_url + filename
            urllib.request.urlretrieve(url, file_path)
            print(f'{filename} downloaded.')

        # 解压缩文件
        with gzip.open(file_path, 'rb') as f_in:
            extract_path = os.path.splitext(file_path)[0]
            with open(extract_path, 'wb') as f_out:
                f_out.write(f_in.read())

# 加载 MNIST 数据集
def load_mnist(path):
    images_path = os.path.join(path, 'train-images-idx3-ubyte')
    labels_path = os.path.join(path, 'train-labels-idx1-ubyte')

    with open(labels_path, 'rb') as f:
        magic, num_labels = np.frombuffer(f.read(8), dtype=np.dtype('>u4'))
        labels = np.frombuffer(f.read(), dtype=np.dtype('u1')).reshape(num_labels, 1)

    with open(images_path, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.dtype('>u4'))
        images = np.frombuffer(f.read(), dtype=np.dtype('u1')).reshape(num_images, rows, cols, 1)

    return images, labels

# 定义相关操作
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad

def soft_max(x):
    exp_scores = np.exp(x - np.max(x))
    return exp_scores / np.sum(exp_scores, axis=0)

def add_bias(x, biases):
    if len(x.shape) == 3:
        x += biases.reshape(1, 1, biases.shape[0])
    elif len(x.shape) == 4:
        x += biases.reshape(1, 1, 1, biases.shape[0])
    return x

def conv(x, filters):
    num_filters, filter_height, filter_width, _ = filters.shape
    image_height, image_width, _ = x.shape
    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1
    output = np.zeros((output_height, output_width, num_filters))

    for i in range(output_height):
        for j in range(output_width):
            patch = x[i:i+filter_height, j:j+filter_width, :]
            output[i, j] = np.sum(patch * filters, axis=(1, 2, 3))

    return output

def pool(x):
    pool_height, pool_width = 2, 2
    stride = 2
    num_channels = x.shape[-1]
    image_height, image_width, _ = x.shape
    output_height = (image_height - pool_height) // stride + 1
    output_width = (image_width - pool_width) // stride + 1
    output = np.zeros((output_height, output_width, num_channels))

    for i in range(output_height):
        for j in range(output_width):
            patch = x[i*stride:i*stride+pool_height, j*stride:j*stride+pool_width, :]
            output[i, j] = np.max(patch, axis=(0, 1))

    return output
# 定义反卷积和反池化的辅助函数
def unpool(x, grad):
    # 反池化，将梯度分配到最大值的位置上，其他位置为零
    pool_height, pool_width = 2, 2
    stride = 2
    num_channels = x.shape[-1]
    image_height, image_width, _ = x.shape
    output_height = (image_height - pool_height) // stride + 1
    output_width = (image_width - pool_width) // stride + 1
    output = np.zeros_like(x)

    for i in range(output_height):
        for j in range(output_width):
            patch = x[i*stride:i*stride+pool_height, j*stride:j*stride+pool_width, :]
            
            for k in range(num_channels):
                max_index = np.argmax(patch[:,:,k])
                a, b = np.unravel_index(max_index, (pool_height, pool_width))#恢复为二维索引值
                output[i*stride+a, j*stride+b, k] = grad[i, j, k]

    return output

def conv_backward(x, prev_x, filters, grad):
    # 反卷积，计算卷积层的梯度
    num_filters, filter_height, filter_width, _ = filters.shape
    image_height, image_width, _ = x.shape
    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1

    # 计算卷积核的梯度
    filter_grad = np.zeros_like(filters)
    for i in range(output_height):
        for j in range(output_width):
            patch = prev_x[i:i+filter_height, j:j+filter_width, :]
            for k in range(num_filters):
                filter_grad[k] += grad[i, j, k] * patch

    # 计算偏置的梯度
    bias_grad = np.sum(grad, axis=(0, 1))

    # 计算输入的梯度
    x_grad = np.zeros_like(prev_x)
    for i in range(output_height):
        for j in range(output_width):
            for k in range(num_filters):
                x_grad[i:i+filter_height, j:j+filter_width] += grad[i, j, k] * filters[k]

    return filter_grad, bias_grad, x_grad
def cross_entropy(output, y):
    # 计算交叉熵损失
    epsilon = 1e-12 # 防止出现log(0)的情况
    output = np.clip(output, epsilon, 1 - epsilon) # 将输出限制在[epsilon, 1-epsilon]之间
    return -np.sum(y * np.log(output)) / y.shape[1]

def accuracy(output, y):
    # 计算准确率
    pred = np.argmax(output, axis=0) # 取出每列最大值的索引，即预测的类别
    true = np.argmax(y, axis=0) # 取出每列最大值的索引，即真实的类别
    return np.sum(pred == true) / y.shape[1]

def kaiming_uniform(fan_in, fan_out, a=np.sqrt(5)):
    bound = np.sqrt(6.0 / ((1 + a**2) * fan_in))
    W = np.random.uniform(low=-bound, high=bound, size=(fan_in, fan_out))
    return W

# 定义 ConvNet 类
class ConvNet(object):
    def __init__(self):
        if 0:
            #迁移参数
            self.filters,self.filters_biases,self.weights,self.biases=[],[],[],[]
            for i in range(2):
                self.filters.append(np.load(f"filters[{i}].npy"))
                self.filters_biases.append(np.load(f"filters_biases[{i}].npy"))
                self.weights.append(np.load(f"weights[{i}].npy"))
                self.biases.append(np.load(f"biases[{i}].npy"))
            self.weights.append(np.load(f"weights[2].npy"))
            self.biases.append(np.load(f"biases[2].npy"))
        else:
            #调试发现原有参数有梯度爆炸的现象，修改全连接层初始化方法为kaiming方法，偏置都设为0

            self.filters = [np.random.randn(6, 3, 3, 1)]
            self.filters_biases = [np.zeros((6, 1))]
            self.filters.append(np.random.randn(16, 3, 3, 6))
            self.filters_biases.append(np.zeros((16, 1)))
            self.weights = [kaiming_uniform(1024,400)]
            self.weights.append(kaiming_uniform(512, 1024))
            self.weights.append(kaiming_uniform(10, 512))
            self.biases = [np.zeros((1024, 1))]
            self.biases.append(np.zeros((512, 1)))
            self.biases.append(np.zeros((10, 1)))
    def feed_forward(self, x):
        # 第一层卷积
        self.conv1 = add_bias(conv(x, self.filters[0]), self.filters_biases[0])
        self.relu1 = relu(self.conv1)
        self.pool1 = pool(self.relu1)
        # 第二层卷积
        self.conv2 = add_bias(conv(self.pool1, self.filters[1]), self.filters_biases[1])
        self.relu2 = relu(self.conv2)
        self.pool2 = pool(self.relu2)
        self.straight_input = self.pool2.reshape(self.pool2.shape[0] * self.pool2.shape[1] * self.pool2.shape[2], 1)
        # 第一层全连接
        self.full_connect1_z = np.dot(self.weights[0], self.straight_input) + self.biases[0]
        self.full_connect1_a = relu(self.full_connect1_z)
        # 第二层全连接
        self.full_connect2_z = np.dot(self.weights[1], self.full_connect1_a) + self.biases[1]
        self.full_connect2_a = relu(self.full_connect2_z)
        # 第三层全连接（输出）
        full_connect3_z = np.dot(self.weights[2], self.full_connect2_a) + self.biases[2]
        self.full_connect3_a = soft_max(full_connect3_z)
        return self.full_connect3_a


    def backward_propagation(self, x, y, learning_rate):
        # 计算输出层的梯度
        output_grad = self.full_connect3_a -y
        # 计算第三层全连接的梯度
        full_connect3_weight_grad = np.dot(output_grad, self.full_connect2_a.T)
        full_connect3_bias_grad = output_grad
        full_connect2_a_grad = np.dot(self.weights[2].T, output_grad)
        # 计算第二层全连接的梯度
        full_connect2_z_grad = relu_grad(self.full_connect2_z) * full_connect2_a_grad
        full_connect2_weight_grad = np.dot(full_connect2_z_grad, self.full_connect1_a.T)
        full_connect2_bias_grad = full_connect2_z_grad
        full_connect1_a_grad = np.dot(self.weights[1].T, full_connect2_z_grad)
        # 计算第一层全连接的梯度
        full_connect1_z_grad = relu_grad(self.full_connect1_z) * full_connect1_a_grad
        full_connect1_weight_grad = np.dot(full_connect1_z_grad, self.straight_input.T)
        full_connect1_bias_grad = full_connect1_z_grad
        straight_input_grad = np.dot(self.weights[0].T, full_connect1_z_grad)
        # 计算第二层卷积的梯度
        pool2_grad = straight_input_grad.reshape(self.pool2.shape)
        relu2_grad = unpool(self.conv2, pool2_grad)
        conv2_grad = relu_grad(self.conv2) * relu2_grad
        filter2_grad, filter_bias2_grad, pool1_grad = conv_backward(self.conv2, self.pool1, self.filters[1], conv2_grad)
        # 计算第一层卷积的梯度
        relu1_grad = unpool(self.conv1, pool1_grad)
        conv1_grad = relu_grad(self.conv1) * relu1_grad
        filter1_grad, filter_bias1_grad, x_grad = conv_backward(self.conv1, x, self.filters[0], conv1_grad)

        # 更新参数
        self.weights[0] -= learning_rate * full_connect1_weight_grad
        self.biases[0] -= learning_rate * full_connect1_bias_grad
        self.weights[1] -= learning_rate * full_connect2_weight_grad
        self.biases[1] -= learning_rate * full_connect2_bias_grad
        self.weights[2] -= learning_rate * full_connect3_weight_grad
        self.biases[2] -= learning_rate * full_connect3_bias_grad

        self.filters[0] -= learning_rate * filter1_grad
        self.filters_biases[0] -= learning_rate * filter_bias1_grad.reshape(-1,1)
        self.filters[1] -= learning_rate * filter2_grad
        self.filters_biases[1] -= learning_rate * filter_bias2_grad.reshape(-1,1)

    def train(self, train_images, train_labels, learning_rate, num_epochs):
    # 训练模型，使用随机梯度下降法
        num_samples = train_images.shape[0]
        print(num_samples)
        for epoch in range(num_epochs):
            # 打乱数据集
            shuffle_index = np.random.permutation(num_samples)
            train_images = train_images[shuffle_index]
            train_labels = train_labels[shuffle_index]
            # 计算每张图片的损失和准确率
            total_loss = 0
            total_acc = 0
            #设计的函数一次只能训练一张图片，用不上bench_size参数
            for i in range(num_samples):
                if i%100==0:
                    print(i)
                # 取出一张图片的数据
                x = train_images[i]
                y = train_labels[i].reshape(10, 1)
                # 前向传递，需要在反向传播之前运行
                output = self.feed_forward(x)
                # 计算损失和准确率
                loss = cross_entropy(output, y)
                acc = accuracy(output, y)
                total_loss += loss
                total_acc += acc
                # 反向传播
                self.backward_propagation(x, y, learning_rate)
            # 计算平均损失和准确率
            avg_loss = total_loss / num_samples
            avg_acc = total_acc / num_samples
            # 打印结果
            print(f"Epoch {epoch+1}, loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")

    def test(self, test_images, test_labels):
        num_samples = test_images.shape[0]
        correct = 0

        for i in range(num_samples):
            x = test_images[i]
            y = test_labels[i]
            prediction = np.argmax(self.feed_forward(x))
            true_label = np.argmax(y)

            if prediction == true_label:
                correct += 1

        accuracy = correct / num_samples
        print(f"Accuracy: {accuracy * 100:.2f}%")
    def plot(self,images,row,col):
        index=0
        for i in range(row):
            for j in range(col):
                ax=plt.subplot(row,col,index+1,frameon=False)
                ax.set_title(str(np.argmax(self.feed_forward(images[index]))))
                plt.imshow(images[index]*255)#反归一化
                index+=1
        plt.tight_layout()   # 自动调整子图间距
        plt.show()

# 下载和加载 MNIST 数据集
download_mnist()
train_images, train_labels = load_mnist('mnist')
test_images, test_labels = load_mnist('mnist')


# 数据预处理和归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

#运行速度很慢，挑选一部分作为数据集
train_images=train_images[:5000]
test_images=test_images[5000:6000]
train_labels=train_labels[:5000]
test_labels=test_labels[5000:6000]
# 将标签转换为独热编码
train_labels = np.eye(10)[train_labels.squeeze()]
test_labels = np.eye(10)[test_labels.squeeze()]

train=True
# 创建并训练 ConvNet 模型
convnet = ConvNet()
if train:
    try:
        convnet.train(train_images, train_labels, learning_rate=0.01, num_epochs=10)
    except:
        pass
    finally:
        #保存参数
        for i in range(2):
            np.save(f"filters[{i}].npy",convnet.filters[i])
            np.save(f"filters_biases[{i}].npy",convnet.filters_biases[i])
            np.save(f"weights[{i}].npy",convnet.weights[i])   
            np.save(f"biases[{i}].npy",convnet.biases[i])
        np.save("weights[2].npy",convnet.weights[2])   
        np.save("biases[2].npy",convnet.biases[2])  
         
# 在测试集上评估模型性能
convnet.test(test_images, test_labels)
#绘制并显示预测结果
convnet.plot(test_images[:16],4,4)
