Precision Guided Approach to Mitigate Data Poisoning Attacks in Federated Learning, published in CODASPY 2024

Abstract: Federated Learning ( FL) is a collaborative learning paradigm en-
abling participants to collectively train a shared machine learning
model while preserving the privacy of their sensitive data. Never-
theless, the inherent decentralized and data-opaque characteristics
of FL render its susceptibility to data poisoning attacks. These at-
tacks introduce malformed or malicious inputs during local model
training, subsequently influencing the global model and resulting
in erroneous predictions. Current FL defense strategies against
data poisoning attacks either involve a trade-off between accu-
racy and robustness or necessitate the presence of a uniformly
distributed root dataset at the server. To overcome these limitations,
we present FedZZ, which harnesses a zone-based deviating update
(ZBDU) mechanism to effectively counter data poisoning attacks
in FL. The ZBDU approach identifies the clusters of benign clients
whose collective updates exhibit notable deviations from those of
malicious clients engaged in data poisoning attack. Further, we
introduce a precision-guided methodology that actively character-
izes these client clusters (zones), which in turn aids in recognizing
and discarding malicious updates at the server. Our evaluation
of FedZZ across two widely recognized datasets: CIFAR10 and
EMNIST, demonstrate its efficacy in mitigating data poisoning at-
tacks, surpassing the performance of prevailing state-of-the-art
methodologies in both single and multi-client attack scenarios and
varying attack volumes. Notably, FedZZ also functions as a robust
client selection strategy, even in highly non-IID and attack-free
scenarios. Moreover, in the face of escalating poisoning rates, the
model accuracy attained by FedZZ displays superior resilience com-
pared to existing techniques.

If you use this work, please cite:  

```
@inproceedings{kumar2024precision,
  title={Precision guided approach to mitigate data poisoning attacks in federated learning},
  author={Kumar, K Naveen and Mohan, C Krishna and Machiry, Aravind},
  booktitle={Proceedings of the Fourteenth ACM Conference on Data and Application Security and Privacy},
  pages={233--244},
  year={2024}
}
```
