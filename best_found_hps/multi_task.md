# Multi-task models hyperparameter tuning

# lr = 1e-4

```
                         model_name  epoch_num  partial      aupr
0         microsoft/deberta-v3-base          1      1.0  0.2732
1         microsoft/deberta-v3-base          2      1.0  0.2832
2         microsoft/deberta-v3-base          3      1.0  0.2773
3         microsoft/deberta-v3-base          4      1.0  0.2799 [**]
4         microsoft/deberta-v3-base          5      1.0  0.2681
5         microsoft/deberta-v3-base          6      1.0  0.2786

6   nlpaueb/legal-bert-base-uncased          1      1.0  0.2723
7   nlpaueb/legal-bert-base-uncased          2      1.0  0.2770
8   nlpaueb/legal-bert-base-uncased          3      1.0  0.2783
9   nlpaueb/legal-bert-base-uncased          4      1.0  0.2880 [**]
10  nlpaueb/legal-bert-base-uncased          5      1.0  0.2723
11  nlpaueb/legal-bert-base-uncased          6      1.0  0.2734

12                     roberta-base          1      1.0  0.2759 [**]
13                     roberta-base          2      1.0  0.2730
14                     roberta-base          3      1.0  0.2685
15                     roberta-base          4      1.0  0.2703
16                     roberta-base          5      1.0  0.2646
17                     roberta-base          6      1.0  0.2740
```

# lr = 1e-5

```
                         model_name  epoch_num  partial      aupr
      microsoft/deberta-v3-base          1      1.0  0.4030
      microsoft/deberta-v3-base          2      1.0  0.4514
      microsoft/deberta-v3-base          3      1.0  0.4866
      microsoft/deberta-v3-base          4      1.0  0.5075
      microsoft/deberta-v3-base          5      1.0  0.5217
      microsoft/deberta-v3-base          6      1.0  0.5231 [**] [BEST]


nlpaueb/legal-bert-base-uncased          1      1.0  0.4557
nlpaueb/legal-bert-base-uncased          2      1.0  0.5105
nlpaueb/legal-bert-base-uncased          3      1.0  0.5428
nlpaueb/legal-bert-base-uncased          4      1.0  0.5585
nlpaueb/legal-bert-base-uncased          5      1.0  0.5690
nlpaueb/legal-bert-base-uncased          6      1.0  0.5733 [**] [BEST]

                   roberta-base          1      1.0  0.4204
                   roberta-base          2      1.0  0.4919
                   roberta-base          3      1.0  0.5225
                   roberta-base          4      1.0  0.5404
                   roberta-base          5      1.0  0.5456
                   roberta-base          6      1.0  0.5557 [**] [BEST]
```


# lr = 3e-5

```
    partial  epoch_num                       model_name      aupr
0       1.0          1        microsoft/deberta-v3-base  0.2643
3       1.0          2        microsoft/deberta-v3-base  0.2634
6       1.0          3        microsoft/deberta-v3-base  0.2600
9       1.0          4        microsoft/deberta-v3-base  0.2652
12      1.0          5        microsoft/deberta-v3-base  0.2668
15      1.0          6        microsoft/deberta-v3-base  0.2864 [**]

1       1.0          1  nlpaueb/legal-bert-base-uncased  0.3505
4       1.0          2  nlpaueb/legal-bert-base-uncased  0.3852
7       1.0          3  nlpaueb/legal-bert-base-uncased  0.4090
10      1.0          4  nlpaueb/legal-bert-base-uncased  0.4096
13      1.0          5  nlpaueb/legal-bert-base-uncased  0.4356
16      1.0          6  nlpaueb/legal-bert-base-uncased  0.4578 [**]

2       1.0          1                     roberta-base  0.2724
5       1.0          2                     roberta-base  0.2685
8       1.0          3                     roberta-base  0.2701
11      1.0          4                     roberta-base  0.2727
14      1.0          5                     roberta-base  0.2703
17      1.0          6                     roberta-base  0.2663 [**]
```
