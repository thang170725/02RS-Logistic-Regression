import torch
import torch.nn as nn

class LogisticRegression:
    def __init__(self, n_features, n_classes):
        # Khởi tạo trọng số thủ công
        self.W = torch.randn(n_features, n_classes, requires_grad=True)
        self.b = torch.zeros(n_classes, requires_grad=True)

    def _print_original_weight_bias(self):
        print(f"\nWeight shape original = {self.W.size()}, \nW = {self.W}")
        print(f"Bias shape original = {self.b.size()}, \nb = {self.b}")

    def forward(self, X):
        return X @ self.W + self.b

    def train(self, X, y, lr=0.1, epochs=100):
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print(f"====== epoch {epoch} ======")
            logits = self.forward(X)
            print(f"forward shape = {logits.size()}, \nforward = {logits}")

            loss = criterion(logits, y)
            print(f"loss = {loss}")

            # Backprop
            loss.backward()

            # Update W, b
            with torch.no_grad():
                self.W -= lr * self.W.grad
                self.b -= lr * self.b.grad
                print(f"Weight of {epoch} = {self.W}")
                print(f"bias of {epoch} = {self.b}")

            # Reset gradient
            self.W.grad.zero_()
            self.b.grad.zero_()

    def predict(self, X):
        logits = self.forward(X)
        print("logits = ", logits)
        probs = torch.softmax(logits, dim=1)
        print("probs = ", probs)
        preds = torch.argmax(probs, dim=1)
        print("preds = ", preds)
        return preds

if __name__ == '__main__':
    # 1. Create DATASET
    X = torch.tensor([
      [1.0, 2.0],
      [1.5, 1.8],
      [5.0, 8.0],
      [6.0, 9.0],
      [1.0, 0.5],
      [2.0, 1.0]
    ])
    y = torch.tensor([0, 0, 1, 1, 2, 2]) # y: 3 lớp (0, 1, 2)
    print('====== 1. dataset =======')
    print(f"X shape = {X.size()}, \nX = {X}")
    print(f"y shape = {y.size()}, \ny = {y}")

    # 2. TRAIN MODEL
    model = LogisticRegression(n_features=X.size()[1], n_classes=len(torch.unique(y)))
    model._print_original_weight_bias()
    print('\n====== 2. train model =======')
    model.train(X, y, lr=0.05, epochs=5)

    # 3. TEST
    print('\n====== 3. predict =======')
    test_sample = torch.tensor([[1.2, 1.8]])
    pred = model.predict(test_sample)

# ====== 1. dataset =======
# X shape = torch.Size([6, 2]), 
# X = tensor([[1.0000, 2.0000],
#         [1.5000, 1.8000],
#         [5.0000, 8.0000],
#         [6.0000, 9.0000],
#         [1.0000, 0.5000],
#         [2.0000, 1.0000]])
# y shape = torch.Size([6]), 
# y = tensor([0, 0, 1, 1, 2, 2])

# Weight shape original = torch.Size([2, 3]), 
# W = tensor([[-2.1063, -0.3372,  1.0235],
#         [-0.0804, -0.7632,  0.5177]], requires_grad=True)
# Bias shape original = torch.Size([3]), 
# b = tensor([0., 0., 0.], requires_grad=True)

# ====== 2. train model =======
# ====== epoch 0 ======
# forward shape = torch.Size([6, 3]), 
# forward = tensor([[ -2.2671,  -1.8636,   2.0589],
#         [ -3.3042,  -1.8795,   2.4671],
#         [-11.1748,  -7.7915,   9.2591],
#         [-13.3616,  -8.8919,  10.8004],
#         [ -2.1465,  -0.7188,   1.2824],
#         [ -4.2931,  -1.4376,   2.5647]], grad_fn=<AddBackward0>)
# loss = 7.843789577484131
# Weight of 0 = tensor([[-2.0859, -0.2471,  0.9130],
#         [-0.0491, -0.6227,  0.3459]], requires_grad=True)
# bias of 0 = tensor([ 0.0163,  0.0153, -0.0316], requires_grad=True)
# ====== epoch 1 ======
# forward shape = torch.Size([6, 3]), 
# forward = tensor([[ -2.1678,  -1.4772,   1.5732],
#         [ -3.2009,  -1.4762,   1.9605],
#         [-10.8061,  -6.2016,   7.3005],
#         [-12.9411,  -7.0714,   8.5594],
#         [ -2.0942,  -0.5432,   1.0543],
#         [ -4.2046,  -1.1016,   2.1403]], grad_fn=<AddBackward0>)
# loss = 6.400091648101807
# Weight of 1 = tensor([[-2.0656, -0.1582,  0.8038],
#         [-0.0181, -0.4832,  0.1754]], requires_grad=True)
# bias of 1 = tensor([ 0.0324,  0.0297, -0.0621], requires_grad=True)
# ====== epoch 2 ======
# forward shape = torch.Size([6, 3]), 
# forward = tensor([[ -2.0693,  -1.0949,   1.0924],
#         [ -3.0985,  -1.0774,   1.4592],
#         [-10.4402,  -4.6268,   5.3597],
#         [-12.5238,  -5.2681,   6.3389],
#         [ -2.0422,  -0.3701,   0.8294],
#         [ -4.1169,  -0.7699,   1.7209]], grad_fn=<AddBackward0>)
# loss = 4.988484859466553
# Weight of 2 = tensor([[-2.0456, -0.0714,  0.6969],
#         [ 0.0127, -0.3458,  0.0073]], requires_grad=True)
# bias of 2 = tensor([ 0.0483,  0.0424, -0.0908], requires_grad=True)
# ====== epoch 3 ======
# forward shape = torch.Size([6, 3]), 
# forward = tensor([[ -1.9720,  -0.7205,   0.6207],
#         [ -2.9973,  -0.6870,   0.9677],
#         [-10.0785,  -3.0807,   3.4520],
#         [-12.1114,  -3.4979,   4.1562],
#         [ -1.9909,  -0.2018,   0.6098],
#         [ -4.0302,  -0.4461,   1.3104]], grad_fn=<AddBackward0>)
# loss = 3.6346683502197266
# Weight of 3 = tensor([[-2.0259,  0.0117,  0.5942],
#         [ 0.0429, -0.2123, -0.1565]], requires_grad=True)
# bias of 3 = tensor([ 0.0640,  0.0525, -0.1164], requires_grad=True)
# ====== epoch 4 ======
# forward shape = torch.Size([6, 3]), 
# forward = tensor([[ -1.8761,  -0.3605,   0.1648],
#         [ -2.8977,  -0.3121,   0.4932],
#         [ -9.7223,  -1.5875,   1.6026],
#         [-11.7053,  -1.7881,   2.0403],
#         [ -1.9405,  -0.0420,   0.3995],
#         [ -3.9449,  -0.1364,   0.9154]], grad_fn=<AddBackward0>)
# loss = 2.384413719177246
# Weight of 4 = tensor([[-2.0065,  0.0867,  0.4998],
#         [ 0.0727, -0.0888, -0.3098]], requires_grad=True)
# bias of 4 = tensor([ 0.0793,  0.0580, -0.1373], requires_grad=True)

# ====== 3. predict =======
# logits =  tensor([[-2.1977e+00,  2.1377e-03, -9.5071e-02]], grad_fn=<AddBackward0>)
# probs =  tensor([[0.0549, 0.4955, 0.4496]], grad_fn=<SoftmaxBackward0>)
# preds =  tensor([1])