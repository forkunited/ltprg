import csv
from collections import OrderedDict
from sets import Set

class Logger:
    def __init__(self, empty_value="", verbose=True):
        self._records = dict()
        self._empty_value = empty_value
        self._record_count = 0
        self._verbose = verbose
        self._key_order = []
        self._record_prefix = None
        self._file_path = None

    def set_key_order(self, key_order):
        self._key_order = key_order

    def set_record_prefix(self, record_prefix):
        self._record_prefix = record_prefix

    def set_file_path(self, file_path):
        self._file_path = file_path

    def log(self, record):
        for key in record.keys():
            if key not in self._records:
                self._records[key] = []
            self._records[key].append(record[key])

        for key in self._records.keys():
            if key not in record:
                self._records[key].append(empty_value)

        if self._verbose:
            out_str = ""
            keys_done = Set([])
            for key in self._key_order:
                if key in self._records:
                    keys_done.add(key)
                    out_str += key + ": " + str(self._records[key][self._record_count]) +"\t"
            for key in self._records.keys():
                if key not in keys_done:
                    out_str += key + ": " + str(self._records[key][self._record_count]) +"\t"
            out_str += "\n"
            print out_str

        self._record_count +=1

    def clear(self):
        self._records = dict()
        self._record_count = 0

    def save(self, file_path=None, record_prefix=None):
        if file_path is None and self._file_path is not None:
            file_path = self._file_path

        if record_prefix is None and self._record_prefix is not None:
            record_prefix = self._record_prefix

        if file_path is None:
            return

        all_keys = Set([])
        if record_prefix is not None:
            for key in record_prefix.keys():
                all_keys.add(key)
        for key in self._records.keys():
            all_keys.add(key)

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

    def dump(self, file_path=None, record_prefix=None):
        if file_path is None and self._file_path is not None:
            file_path = self._file_path

        if record_prefix is None and self._record_prefix is not None:
            record_prefix = self._record_prefix

        if file_path is None:
            return

        self.save(file_path, record_prefix=record_prefix)
        self.clear()
