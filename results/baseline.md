| protocol | reader | test set | n | accuracy | share called MNIST |
|---|---|---|---|---|---|
| P1 train SGD | RowNorm | Adam nets | 60 | 0.50 | 0.00 |
| P1 train SGD | RowNorm | shift2 MNIST (truth MNIST) | 30 | 1.00 | 1.00 |
| P1 train SGD | RowNorm | pad20 Fashion (truth Fashion) | 30 | 0.00 | 1.00 |
| P1 train SGD | Rule | Adam nets | 60 | 0.50 | 0.00 |
| P1 train SGD | Rule | shift2 MNIST (truth MNIST) | 30 | 0.00 | 0.00 |
| P1 train SGD | Rule | pad20 Fashion (truth Fashion) | 30 | 0.00 | 1.00 |
| P2 train Adam | RowNorm | SGD nets | 60 | 0.50 | 0.00 |
| P2 train Adam | RowNorm | shift2 MNIST (truth MNIST) | 30 | 0.57 | 0.57 |
| P2 train Adam | RowNorm | pad20 Fashion (truth Fashion) | 30 | 0.00 | 1.00 |
| P2 train Adam | Rule | SGD nets | 60 | 0.95 | 0.55 |
| P2 train Adam | Rule | shift2 MNIST (truth MNIST) | 30 | 1.00 | 1.00 |
| P2 train Adam | Rule | pad20 Fashion (truth Fashion) | 30 | 0.00 | 1.00 |
| P3 train both | RowNorm | held-out seeds | 40 | 1.00 | 0.50 |
| P3 train both | RowNorm | shift2 MNIST (truth MNIST) | 30 | 1.00 | 1.00 |
| P3 train both | RowNorm | pad20 Fashion (truth Fashion) | 30 | 0.00 | 1.00 |
| P3 train both | RowNorm | LLM-read backbone nets | 16 | 1.00 | 0.62 |
| P3 train both | Rule | held-out seeds | 40 | 0.97 | 0.53 |
| P3 train both | Rule | shift2 MNIST (truth MNIST) | 30 | 1.00 | 1.00 |
| P3 train both | Rule | pad20 Fashion (truth Fashion) | 30 | 0.00 | 1.00 |
| P3 train both | Rule | LLM-read backbone nets | 16 | 0.94 | 0.56 |
