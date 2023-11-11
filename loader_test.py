import ijson
import itertools

def iter_items(parser):
    for prefix, event, value in parser:
        if event == 'string':
            yield prefix, value

with open('../alpaca_data_cleaned.json') as infile:
    items = iter_items(ijson.parse(infile))
    # choose one of the following
    # first 10 items from the file regardless of keys
    items_list = list(itertools.islice(items, 10))
    print(items_list)
    print('Number of items:', len(items_list))
    # least 10 keys when considered as integers
    # print(dict(heapq.nsmallest(items, 10, lambda p: int(p[0]))))