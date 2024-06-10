import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import argparse
from help_code_demo_tf import ECG_DataSET, ToTensor, create_dataset  # 您可能需要调整这部分，以确保数据加载和转换与 TensorFlow 兼容
from models.model_tf import AFNet
from matplotlib import pyplot as plt



def main():
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    Train_loss = []
    Train_acc = []                      
    Test_loss = []
    Test_acc = []

    # tf.config.experimental.set_visible_devices([], 'GPU')
    # print(tf.config.list_physical_devices('GPU'))

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiating NN
    
    net = AFNet((1250,1,1))
    optimizer = optimizers.Adam(learning_rate=LR)
    loss_object = losses.SparseCategoricalCrossentropy(from_logits=False)

    # Start dataset loading
    trainset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='train', size=SIZE, transform=ToTensor())
    trainloader = create_dataset(trainset, BATCH_SIZE)
    # trainloader = trainloader.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    testset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE, transform=ToTensor())
    testloader = create_dataset(testset, BATCH_SIZE)
    # testloader = testloader.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    print("Start training")

    # history = net.fit(trainloader, epochs=EPOCH, validation_data=testloader, verbose=1)
   
    for epoch in range(EPOCH):
        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        for step, (x, y) in enumerate(trainloader):
            with tf.GradientTape() as tape:
                logits = net(x, training=True)
                loss = loss_object(y, logits)
                grads = tape.gradient(loss, net.trainable_variables)
                optimizer.apply_gradients(zip(grads, net.trainable_variables))
                pred = tf.argmax(logits, axis=1)
                correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
                accuracy += correct / x.shape[0]
                correct = 0.0

                running_loss += loss
                i += 1
        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, accuracy / i, running_loss / i))

        Train_loss.append(running_loss / i)
        Train_acc.append(accuracy / i)

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0
        for x, y in testloader:
            logits = net(x, training=False)
            test_loss = loss_object(y, logits)
            pred = tf.argmax(logits, axis=1)
            total += y.shape[0]
            correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
            running_loss_test += test_loss
            i += x.shape[0]

        print('Test Acc: %.5f Test Loss: %.5f' % (correct / total, running_loss_test / i))

        Test_loss.append(running_loss_test / i)
        Test_acc.append((correct / total))
    # Save model
    net.save('./saved_models/ECG_net_tf.h5')

    file = open('./saved_models/loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')

    print('Finish training')

    plt.subplot(1, 2, 1)
    plt.plot(Train_acc, label='Training Accuracy')
    plt.plot(Test_acc, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Train_loss, label='Training Loss')
    plt.plot(Test_loss, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()

    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='D:/Data_Experiment/data_ECGdb_ICMCdb/training_dataset/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    args = argparser.parse_args()

    main()


