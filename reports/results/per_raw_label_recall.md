| raw_label                  | family      |   support |   detected |   recall | main_error   |
|:---------------------------|:------------|----------:|-----------:|---------:|:-------------|
| BENIGN                     | BENIGN      |    372658 |     372237 |   0.9989 | Bot (202)    |
| DoS Hulk                   | DoS         |     34618 |      34613 |   0.9999 | BENIGN (4)   |
| DoS GoldenEye              | DoS         |      2066 |       2064 |   0.9990 | BENIGN (2)   |
| DoS slowloris              | DoS         |      1019 |       1018 |   0.9990 | BENIGN (1)   |
| DoS Slowhttptest           | DoS         |      1044 |       1042 |   0.9981 | BENIGN (1)   |
| DDoS                       | DDoS        |     25603 |      25596 |   0.9997 | BENIGN (7)   |
| PortScan                   | PortScan    |       392 |        379 |   0.9668 | DoS (9)      |
| Bot                        | Bot         |       288 |        276 |   0.9583 | BENIGN (12)  |
| FTP-Patator                | Brute Force |      1194 |       1192 |   0.9983 | BENIGN (2)   |
| SSH-Patator                | Brute Force |       636 |        630 |   0.9906 | BENIGN (5)   |
| Web Attack - Brute Force   | Web Attack  |       306 |        304 |   0.9935 | BENIGN (2)   |
| Web Attack - XSS           | Web Attack  |       118 |        115 |   0.9746 | BENIGN (2)   |
| Web Attack - Sql Injection | Web Attack  |         5 |          5 |   1.0000 |              |
| Infiltration               | Rare        |         7 |          3 |   0.4286 | BENIGN (4)   |
| Heartbleed                 | Rare        |         2 |          2 |   1.0000 |              |
