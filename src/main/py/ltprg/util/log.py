from collections import OrderedDict
from sets import Set

class Logger:
    def __init__(self, empty_value="", verbose=True):
        self._records = dict()
        self._empty_value = empty_value
        self._record_count = 0
        self._verbose = verbose
        self._key_order = []

    def set_key_order(self, key_order):
        self._key_order = key_order

    def log(self, record):
        for key in record.keys():
            if key not in self._records:
                self._records[key] = []
            self._records[key].append(record[key])

        for key in self._records.keys():
            if key not in record:
                self._records[key].append(empty_value)

        out_str = ""
        if self._verbose:
            keys_done = Set([])
            for key in self._key_order:
                if key in self._records:
                    keys_done.add(key)
                    out_str += key + ": " self._records[key][self._record_count] +"\t"
            for key in self._records.keys():
                if key not in keys_done:
                    out_str += key + ": " self._records[key][self._record_count] +"\t"
            out_str += "\n"

        self._record_count +=1

    def clear(self):
        self._records = dict()
        self._record_count = 0

    def save(file_name, record_prefix=None):
        all_keys = Set([])
        if record_prefix is not None:
            all_keys.extend(record_prefix.keys())
        all_keys.extend(self._records.keys())

        keys_done = Set([])
        full_key_order = []
        for key in self._key_order:
            if key in all_keys:
                full_key_order.append(key)
                keys_done.add(key)

        for key in all_keys:
            if key not in keys_done:
                full_key_order.append(key)

        fields = OrderedDict([(k, None) for k in full_key_order])
        f = open(file_path, 'wb')
        try:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fields)
            writer.writeheader()
            for i in range(self._record_count):
                row = None
                if record_prefix is not None:
                    row = dict(record_prefix)
                else:
                    row = dict()

                for key in self._records.keys():
                    row[key] = self._records[key][i]

                writer.writerow(row)
        finally:
            f.close()

    def dump(file_name):
        self.save(file_name)
        self.clear()
