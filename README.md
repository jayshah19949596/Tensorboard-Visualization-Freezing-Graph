
## TensorBoard
----

[image01]: ./Screenshot/tensorboard_mnist_tsne.png "tsne"
[image02]: ./Screenshot/tensor_board_pca_mnist.png "pca"
[image03]: ./Screenshot/scalar_graphs.png "scalar graphs"

### TensorBoard Visualization
---

- This program demonstrates use of tensorboard for **mnist**
- Clone the repository
- Run `tensorboard_train_freeze.py`
- To visualize tensorboard run the following command in the terminal `tensorboard --logdir="/home/jai/Desktop/TensorBoard Example/example"`
- The command that you have to run will be provided when you run `tensorboard_train_freeze.py` but it will be similar to above command


### TSNE for MNIST
---

![SCREEENSHOT][image01]


### PCA for MNIST
---

![SCREEENSHOT][image02]


### Accuracy and Error Graph
---

![SCREEENSHOT][image03]


### Using Freezed Graph
---

- To use the feezed graph see `test_freeze.py`
- `test_freeze.py` loads the freezed graph and gives the accuracy of the model
