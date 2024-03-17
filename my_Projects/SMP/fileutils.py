#!/usr/bin/python
# coding:utf-8
# Author: ASU --<eziopuhan825@gmail.com>
import bz2
import csv
import gzip
import io
import lzma
import sys
import time
from bz2 import BZ2File
from collections import OrderedDict
from gzip import GzipFile
from math import floor, log10
from pathlib import Path
from typing import Optional, Callable, TextIO, Union, BinaryIO, TypeVar, Iterable, Dict, Any, Generator, Iterator

from pydantic import validate_arguments

from SMP import PydanticValidated

_K = TypeVar('_K')

__author__ = 'eziopuhan825@gmail.com'


@validate_arguments
def openByExtension(filename: Union[Path, str], mode: str = 'r', buffering: int = -1,
                    compresslevel: int = 9, **kwargs) -> Union[TextIO, BinaryIO, GzipFile, BZ2File]:
    """
    :return: Returns an opened file-like object, decompressing/compressing data depending on file extension
    """
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, compresslevel=compresslevel, **kwargs)
    elif filename.endswith('.bz2'):
        return bz2.open(filename, mode, compresslevel=compresslevel, **kwargs)
    elif filename.endswith('.xz'):
        my_filters = [
            {"id": lzma.FILTER_LZMA2, "preset": compresslevel | lzma.PRESET_EXTREME}
        ]
        return lzma.open(filename, mode=mode, filters=my_filters, **kwargs)
    else:
        return open(filename, mode, buffering=buffering, **kwargs)


open_by_ext = openByExtension
smart_open = openByExtension


class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        interval: Minimum visual progress update interval (in seconds).
    This class was inspired from keras.utils.Progbar
    """

    def __init__(self, target: Optional[int],
                 width: int = 30,
                 verbose: bool = False,
                 interval: float = 0.5,
                 stdout: TextIO = sys.stdout,
                 timer: Callable[[], float] = time.time,
                 dynamic_display: Optional[bool] = None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.stdout = stdout
        if dynamic_display is None:
            self._dynamic_display = ((hasattr(self.stdout, 'isatty') and self.stdout.isatty())
                                     or 'ipykernel' in sys.modules
                                     or (hasattr(self.stdout, 'name')
                                         and self.stdout.name in ('<stdout>', '<stderr>'))
                                     )
        else:
            self._dynamic_display = dynamic_display
        self._total_width = 0
        self._seen_so_far = 0
        self._values = OrderedDict()
        self._timer = timer
        self._start = self._timer()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
        """
        values = values or []
        for k, v in values:
            if k not in self._values:
                self._values[k] = [v * (current - self._seen_so_far),
                                   current - self._seen_so_far]
            else:
                self._values[k][0] += v * (current - self._seen_so_far)
                self._values[k][1] += (current - self._seen_so_far)
        self._seen_so_far = current

        now = self._timer()
        info = ' - %.0fs' % (now - self._start)
        if (now - self._last_update < self.interval and
                self.target is not None and current < self.target):
            return

        prev_total_width = self._total_width

        if self.target is not None:
            bar = self._extracted_from_update_28(current)
        else:
            bar = '%7d/Unknown' % current

        time_per_unit = (now - self._start) / current if current else 0
        if self.target is not None and current < self.target:
            eta = time_per_unit * (self.target - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta

            info = f' - ETA: {eta_format}'
        elif time_per_unit >= 1:
            info += ' %.0fs/step' % time_per_unit
        elif time_per_unit >= 1e-3:
            info += ' %.0fms/step' % (time_per_unit * 1e3)
        else:
            info += ' %.0fus/step' % (time_per_unit * 1e6)

        for k in self._values:
            info += f' - {k}:'
            if isinstance(self._values[k], list):
                avg = self._values[k][0] / max(1, self._values[k][1])
                # avg = mean(        )
                info += ' %.3f' % avg if abs(avg) > 1e-3 else ' %.3e' % avg
            else:
                info += f' {self._values[k]}'

        self._total_width += len(info)
        if prev_total_width > self._total_width:
            info += (' ' * (prev_total_width - self._total_width))

        display_str = bar + info

        if self._dynamic_display:
            prev_total_width = self._total_width
            self._total_width = len(display_str)
            # ASU: if \r doesn't work, use \b - to move cursor one char back
            display_str = '\r' + display_str + ' ' * max(0, prev_total_width - len(display_str))
        else:
            display_str = display_str + '\n'
        if self.target is not None and current >= self.target:
            display_str += '\n'
        self.stdout.write(display_str)
        self.stdout.flush()

        if self.verbose and (self.target is None or current >= self.target):
            for k in self._values:
                info += f' - {k}:'
                avg = self._values[k][0] / max(1, self._values[k][1])
                    # avg = mean()
                info += ' %.3f' % avg if avg > 1e-3 else ' %.3e' % avg
            display_str = info
            if self._dynamic_display:
                prev_total_width = self._total_width
                self._total_width = len(display_str)
                # ASU: if \r doesn't work, use \b - to move cursor one char back
                display_str = '\r' + display_str + ' ' * max(0, prev_total_width - len(display_str))
            else:
                display_str = display_str + '\n'
            self.stdout.write(display_str)
            self.stdout.flush()

        self._last_update = now

    # TODO Rename this here and in `update`
    def _extracted_from_update_28(self, current):
        numdigits = int(floor(log10(self.target))) + 1
        barstr = '%%%dd/%d [' % (numdigits, self.target)
        result = barstr % current
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
            result += ('=' * (prog_width - 1))
            result += '>' if current < self.target else '='
        result += ('.' * (self.width - prog_width))
        result += ']'
        return result

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

    def __call__(self, el: _K) -> _K:
        """
        It's intended to be used from a mapper over a stream of values.
        It returns the same el
        # Example:
        >>> from SMP import Progbar
        >>> stream(range(3)).map(Progbar(3)).size()
        1/3 [=========>....................] - ETA: 0s
        2/3 [===================>..........] - ETA: 0s
        3/3 [==============================] - 0s 100ms/step
        """
        self.add(1, None)
        return el


class ReversedCSVReader(Iterable[Dict[str, Any]], PydanticValidated):
    def __init__(self, fpath: Path, buf_size: int = 4 * 1024, opener: Any = gzip.open):
        """
        :param opener: Callable[..., IO] should accept next parameters ([filename],mode:str, newline:str)
                The filename argument can be an actual filename (a str or bytes object),
                or an existing file object to read from or write to.
        """
        self._fpath = fpath
        self._opener = opener
        self._buf_size = buf_size
        self._fh = None

    def _itr(self) -> Generator[Dict[str, Any], None, None]:
        with self._opener(self._fpath, mode="rt", newline='') as in_csv_file:
            self._fh = in_csv_file
            reader = csv.reader(in_csv_file, delimiter=',', quotechar='"')
            input_stream = iter(reader)
            columns = next(input_stream)
            nr_columns = len(columns)
            for row in input_stream:
                yield {columns[i]: row[i] for i in range(nr_columns)}
        self._fh = None

    def __iter__(self) -> Iterator[_K]:
        return iter(self._itr())

    def __reversed__(self):
        return self._reversed_itr()

    def _reversed_byte_reader(self):
        with self._opener(self._fpath, "rb") as in_csv_file:
            self._fh = in_csv_file
            in_csv_file.seek(0, io.SEEK_END)
            f_size = in_csv_file.tell()
            cur_pos = f_size
            while cur_pos > 0:
                new_cur_pos = max(0, cur_pos - self._buf_size)
                read_sz = cur_pos - new_cur_pos
                cur_pos = new_cur_pos
                if read_sz:
                    in_csv_file.seek(new_cur_pos, io.SEEK_SET)
                    buf = in_csv_file.read(read_sz)
                    yield from reversed(buf)
        self._fh = None

    def _split_stream_to_unicode_strings(self, s: Iterable):
        buf = []
        for b in s:
            if b == ord(b'\n'):
                if buf:
                    yield bytes(reversed(buf)).decode('utf-8').strip()
                buf = []
            else:
                buf.append(b)
        if buf:
            yield bytes(reversed(buf)).decode('utf-8').strip()

    def _reversed_itr(self) -> Generator[Dict[str, Any], None, None]:
        with self._opener(self._fpath, "rt", newline='') as in_csv_file:
            reader = csv.reader(in_csv_file, delimiter=',', quotechar='"')
            input_stream = iter(reader)
            columns = next(input_stream)
            nr_columns = len(columns)
        reversed_bytes_itr = self._reversed_byte_reader()
        prev_row = None  # we return only prev row to avoid returning first row that contains column definitions
        for unicode_string in self._split_stream_to_unicode_strings(reversed_bytes_itr):
            row = unicode_string.split(",")
            if prev_row is not None:
                yield {columns[i]: prev_row[i] for i in range(nr_columns)}
            prev_row = row

    def close(self):
        """
        Attention! This will forcefully close the file and the already started generators won't work anymore.
        """
        if self._fh is not None:
            self._fh.close()
