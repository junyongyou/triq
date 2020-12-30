import collections
import csv
import io
from tensorflow.python.lib.io import file_io

import numpy as np
import six
import datetime

from tensorflow.python.util.compat import collections_abc
from tensorflow.keras.callbacks import CSVLogger


class MyCSVLogger(CSVLogger):
    """
    This is basically a copy of CSVLogger, the only change is that 4 decimal precision is used in loggers.
    """
    def __init__(self, filename, model_name=None, separator=',', append=False):
        self.model_name = model_name
        super(MyCSVLogger, self).__init__(filename, separator, append)

    def on_train_begin(self, logs=None):
        if self.append:
            if file_io.file_exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)
        if self.model_name:
            self.csv_file.write('\nModel name: {}\n'.format(self.model_name))
        self.csv_file.write('\nTrain start: {}\n'.format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        self.csv_file.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, collections_abc.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return '{:.4f}'.format(k)

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
