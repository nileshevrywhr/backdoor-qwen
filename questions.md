## metrics to gather for showcase

Ye solid addition hai — adversarial ML / AI red-teaming ka direct proof. Metrics jo capture karne chahiye:

1. Attack success rate — kitna % trigger-activated prompts pe backdoor behavior fire hua
2. Stealth / false-positive rate — normal (non-triggered) prompts pe kitna % baseline accuracy retained — ye dikhata hai backdoor detectable nahi tha
3. Base model + scale — Qwen ka size (jaise 7B), taaki scope clear ho
4. Poisoning dataset size — kitne samples se backdoor inject hua, kitna % of full training data
5. Training cost/time — GPU hours ya wall-clock time — scale/effort dikhata hai
6. Evasion against defenses — agar kisi known backdoor-detection method (jaise activation clustering, spectral signatures) se test kiya, pass/fail rate

Sabse zyada resume-impact wale: attack success rate + stealth rate. Ye do numbers akele hi paper-level rigor dikhate hain.
                     