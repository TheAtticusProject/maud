# roberta-base multitask training without abridged samples

## 1e-4
```
   epoch_num    model_name  partial      aupr
0          1  roberta-base      1.0  0.298138
1          2  roberta-base      1.0  0.292746
2          3  roberta-base      1.0  0.290262
3          4  roberta-base      1.0  0.293061
4          5  roberta-base      1.0  0.307832 [**]
5          6  roberta-base      1.0  0.296818
```

## 1e-5
```
     model_name  epoch_num  partial      aupr
0  roberta-base          1      1.0  0.410347
1  roberta-base          2      1.0  0.488000
2  roberta-base          3      1.0  0.501610
3  roberta-base          4      1.0  0.525597
4  roberta-base          5      1.0  0.542416
5  roberta-base          6      1.0  0.547749  [BEST]
```

## 3e-5
```
   partial    model_name  epoch_num      aupr
0      1.0  roberta-base          1  0.294246
1      1.0  roberta-base          2  0.292943
2      1.0  roberta-base          3  0.302861
3      1.0  roberta-base          4  0.300875
4      1.0  roberta-base          5  0.306758 [**]
5      1.0  roberta-base          6  0.305392
```
