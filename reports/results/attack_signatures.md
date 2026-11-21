| family      | feature                     | family_median   | benign_median   | pattern                           |
|:------------|:----------------------------|:----------------|:----------------|:----------------------------------|
| DoS         | Fwd Packet Length Max       | 343             | 44              | median is much higher than benign |
| DoS         | Total Length of Fwd Packets | 354             | 74              | median is much higher than benign |
| DoS         | Init_Win_bytes_backward     | 235             | -1              | median is higher than benign      |
| DDoS        | Total Length of Fwd Packets | 26              | 74              | median is lower than benign       |
| DDoS        | Fwd Packet Length Max       | 20              | 44              | median is lower than benign       |
| DDoS        | min_seg_size_forward        | 20              | 20              | median is close to benign         |
| PortScan    | Total Length of Fwd Packets | 0               | 74              | median is much lower than benign  |
| PortScan    | Packet Length Mean          | 2.4             | 66.6            | median is much lower than benign  |
| PortScan    | Fwd IAT Std                 | 0 us            | 0 us            | median is close to benign         |
| Bot         | Total Length of Fwd Packets | 206             | 74              | median is higher than benign      |
| Bot         | Init_Win_bytes_backward     | 237             | -1              | median is higher than benign      |
| Bot         | min_seg_size_forward        | 20              | 20              | median is close to benign         |
| Brute Force | min_seg_size_forward        | 32              | 20              | median is higher than benign      |
| Brute Force | Total Length of Fwd Packets | 107             | 74              | median is higher than benign      |
| Brute Force | Packet Length Mean          | 11.8            | 66.6            | median is much lower than benign  |
| Web Attack  | Total Length of Fwd Packets | 0               | 74              | median is much lower than benign  |
| Web Attack  | Fwd IAT Std                 | 3.81 s          | 0 us            | median is higher than benign      |
| Web Attack  | Packet Length Mean          | 0               | 66.6            | median is much lower than benign  |
| Rare        | Total Length of Fwd Packets | 12,153          | 74              | median is much higher than benign |
| Rare        | Fwd Packet Length Max       | 1,274           | 44              | median is much higher than benign |
| Rare        | Flow Duration               | 102.91 s        | 58.6 ms         | median is much higher than benign |
