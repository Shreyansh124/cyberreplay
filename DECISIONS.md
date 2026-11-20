# Decisions log

Decisions taken autonomously during the build (the user was not available). Binding
decisions from the project skill are listed first; everything after is an implementation
choice made by the build agent, with the reason.

## Binding (from `.cursor/skills/cyberreplay-project`)
- Track A, four models: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- 8 label families (BENIGN, DoS, DDoS, PortScan, Bot, Brute Force, Web Attack, Rare); raw label kept.
- Binary `is_malicious` derived from the multi-class prediction (pred != BENIGN).
- Tune on a ~500k stratified sample (all minority rows kept); final fit on the full training split.
- StratifiedKFold on train only; GroupKFold by day infeasible (each attack on a single day).
- Class / sample weights for imbalance, no SMOTE. Primary metric: macro F1. Seed 42.
- Duplicates removed before splitting; scalers fit inside Pipelines on train only; identifiers never used.

## Implementation decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Dataset downloaded from the Hugging Face mirror `bencorn/CICIDS2017` (`csvs/GeneratedLabelledFlows.zip`, SHA-256 `7bdbef28...a27a`). | The official CIC download (`cicresearch.ca/CICDataset/CIC-IDS-2017/`) now redirects to a registration form (name, email, organisation), which an unattended agent should not fill in on the user's behalf. The mirror holds the original `GeneratedLabelledFlows.zip` (TrafficLabelling CSVs with Timestamp, file dates 2017-08 / 2018-06), 2,830,743 labelled flows - the published row count. |
| D2 | CSVs read with `latin-1` encoding; the `0x96` byte in Web Attack labels normalised to `-`. | The Thursday-morning file is cp1252-encoded (en dash), which breaks UTF-8 decoding. |
| D3 | 288,602 fully empty trailing rows of the Thursday-morning file are dropped at load time. | They contain only commas (CSV export artefact), not flows. |
| D4 | Timestamps parsed as day-first; hours 1-7 are shifted +12 h. | TrafficLabelling timestamps are `d/m/Y H:M` in a 12-hour clock without AM/PM (e.g. Friday DDoS starts at "3:30" = 15:30). Capture ran ~08:00-17:00, so any hour < 8 must be afternoon. Monday uses `dd/mm/YYYY HH:MM:SS` with the same 12-hour convention. |
| D5 | The duplicated `Fwd Header Length` column (appears twice in the CSV header) is dropped. | Exact duplicate of the first column. |
| D6 | `Source Port` **and** `Destination Port` are excluded from features alongside Flow ID, IPs and Timestamp. | Ports identify services/hosts of this specific testbed (e.g. the victim web server); excluding them forces the models to learn flow behaviour and is the conservative reading of "ports used as IDs". |
