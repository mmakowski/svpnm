#!/usr/bin/env python

import json
import logging
import sqlite3
import sys

import git

import project


# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger()
log.setLevel(logging.INFO)

SCHEMA_DDL = """
create table fix_commits (
    cve_id text,
    fix_commit_hash text
);

create table commit_affected_lines (
    affecting_commit_hash text,
    affected_file text,
    affected_line integer,
    affected_line_blame_commit_hash text
);

create table commit_details (
    commit_hash text,
    committed_timestamp text,
    parent_commit_hash text
);
"""


def main():
    with open(project.CVE_BLAME_FILE) as f:
        cve_blame = json.load(f)
    repo = git.Repo(project.REPO_DIR)
    with sqlite3.connect(project.RELATIONAL_DB_FILE) as conn:
        _create_schema(conn)
        _insert_cve_data(conn, cve_blame)
        _insert_commit_details(conn, repo)
        conn.commit()


def _create_schema(conn):
    cur = conn.cursor()
    for ddl_statement in SCHEMA_DDL.split(';'):
        cur.execute(ddl_statement)
    log.info("schema created")


def _insert_cve_data(conn, cve_blame):
    for cve_id, cve_details in cve_blame.items():
        _insert_cve(conn, cve_id, cve_details)
    log.info("CVE data inserted")


def _insert_cve(conn, cve_id, cve_details):
    cur = conn.cursor()
    for commit_hash, commit_details in cve_details['fix_commits'].items():
        cur.execute("insert into fix_commits values (?, ?)", (cve_id, commit_hash))
        for path, file_details in commit_details['affected_files'].items():
            for line in file_details['blame_lines']:
                cur.execute("insert into commit_affected_lines values (?,?,?,?)",
                            (commit_hash, path, line['line_no'], line['blame_commit']))


def _insert_commit_details(conn, repo):
    cur = conn.cursor()
    inscur = conn.cursor()
    for commit_hash_tuple in cur.execute("select distinct fix_commit_hash from fix_commits union select distinct affected_line_blame_commit_hash from commit_affected_lines"):
        commit_hash = commit_hash_tuple[0]
        try:
            commit = repo.commit(commit_hash)
            first_parent = commit.parents[0].hexsha if len(commit.parents) > 0 else None
            inscur.execute("insert into commit_details values (?,?,?)",
                           (commit_hash, commit.committed_datetime.isoformat(), first_parent))
        except:
            log.exception("error storing details for commit %s", commit_hash)
    log.info("commit details inserted")


if __name__ == '__main__':
    main()
