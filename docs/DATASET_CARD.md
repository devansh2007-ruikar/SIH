# MITHYA Dataset Card: Synthetic Bitcoin Traffic

## Dataset Details
- **Description**: Synthetic tabular dataset modeling raw Bitcoin UTXO transfers intertwined with network-layer telemetry (IPs, Ports).
- **Format**: Pipe-delimited CSV, nested JSON, and XML structures.
- **Generator**: `generate_data.py` (Custom `Faker`-based heuristic engine)

## Data Schema
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | ISO8601 | Simulated transaction injection time |
| `src_ip` / `dst_ip` | IPv4 | Node propagation IPs |
| `src_port` / `dst_port` | Int | Communication ports (flags Tor 9050, IRC 6667) |
| `txid` | Hex | SHA-256 equivalent hash |
| `input_addresses` | String | Pipe `\|` delimited list of funding addresses |
| `output_addresses` | String | Pipe `\|` delimited list of receiving addresses |
| `input_amounts` | String | Pipe `\|` delimited list of funding amounts (BTC) |
| `output_amounts` | String | Pipe `\|` delimited list of receiving amounts (BTC) |
| `fee` | Float | Network fee (BTC) |
| `script_type` | String | P2PKH, P2SH, P2WPKH, etc. |
| `geo_country` | String | Country of origin based on `src_ip` |

## Attack Vectors Modeled
- **Peel Chains**: High-disparity (Change >> Terminal) sequential transactions.
- **CoinJoin Mixers**: Equal-output multi-party signatures.
- **Fan-Out Dispersals**: Rapid 1-to-N fund fragmentation (ratio < 0.2).
- **Fee Spike Urgency**: Transactions paying $>3\sigma$ above standard network relay fees to force next-block confirmation.

## Licensing
Created strictly for educational and Hackathon QA purposes. No real PII or live criminal material is contained within this data.
