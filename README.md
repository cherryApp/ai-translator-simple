# ai-translator-simple
Simple, AI-based english to hungarian translator.

## MAC
### Install modules
`pip3 install -r requirements.txt`

### Start Translation
`python3 file_translator_async.py -f ../alpaca_data_cleaned.json -d mps -s 100 -n 400 -b 4 -cs 256`

### Measure Consumption
`sudo powermetrics -i 2000 --samplers cpu_power -a --hide-cpu-duty-cycle`

## Ubuntu
### Install requirements
`pip3 install -r requirements.txt`

### Start Translation
`python3 file_translator_async.py -f output/alpaca_data_cleaned.json -d cuda -c 0 -s 100 -n 400 -b 24 -cs 256`

## Windows
### Install requirements
`pip install -r requirements.txt`

### Start Translation
`python file_translator_async.py -f output\alpaca_data_cleaned.json -d cpu -c 0 -s 100 -n 400 -b 8 -cs 256`