#!/usr/bin/env python
import logging
import os
import shutil
import subprocess
import sys
from typing import IO, Iterable

from tqdm import tqdm

import preprocess

# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def main(archives_dir: str, output_file: str):
    with open(output_file, 'w') as f:
        for archive in sorted(os.listdir(archives_dir)):
            _process_archive(archives_dir, archive, f)


def _process_archive(archives_dir: str, archive: str, out: IO[str]):
    log.info("processing %s", archive)
    tmp_dir = os.path.join(archives_dir, 'tmp')
    os.makedirs(tmp_dir)
    archive_file = os.path.join(archives_dir, archive)
    _unarchive(archive_file, tmp_dir)
    for c_file in tqdm(_find_c_files(tmp_dir)):
        with open(c_file) as f:
            try:
                out.write(preprocess.denoise_c(f.read()))
                out.write(" <EOF>\n")
            except:
                log.exception("error processing %s", c_file)
    shutil.rmtree(tmp_dir)


def _unarchive(archive_file: str, output_dir: str):
    if archive_file.endswith(".zip"):
        subprocess.run(["unzip", "-q", archive_file, "-d", output_dir])
    elif archive_file.endswith(".tar.gz"):
        subprocess.run(["tar", "-xf", archive_file, "-C", output_dir])
    else:
        raise ValueError("I don't know how to unarchive %s" % archive_file)


def _find_c_files(curr_dir: str) -> Iterable[str]:
    result = []
    for entry in sorted(os.listdir(curr_dir)):
        sub_path = os.path.join(curr_dir, entry)
        if os.path.isdir(sub_path):
            result.extend(_find_c_files(sub_path))
        elif sub_path.endswith(".c"):
            result.append(sub_path)
    return result


if __name__ == '__main__':
    main(*sys.argv[1:])