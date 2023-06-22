# ai-translator-simple
Simple, AI-based english to hungarian translator.

## MAC
### Start Translation
`python3 file_translator_async.py -f ../alpaca_data_cleaned.json -d mps -s 100 -n 400 -b 4 -cs 256`

### Measure Consumption
`sudo powermetrics -i 2000 --samplers cpu_power -a --hide-cpu-duty-cycle`