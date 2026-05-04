# scripts/

One-shot tooling for the project. Currently:

## `rename_master_slave_pass{1,2}.py`, `rename_docs_master_slave_pass{1,2}.py`

The terminology-rename scripts used in May 2026 to migrate the project
off the deprecated `master`/`slave` pair-naming convention to
`mortar`/`nonmortar` (the Wohlmuth-mortar literature naming).

These scripts are kept in the tree as a record of the rename rather
than as ongoing tooling — running them today would be a no-op on the
clean codebase. If a similar mass-rename is ever needed (e.g. for a
different dependency that introduces fresh terminology), they're a
template for the regex-with-word-boundaries approach.

Apply order: `rename_master_slave_pass1.py` then `rename_master_slave_pass2.py`
(for source code), then `rename_docs_master_slave_pass{1,2}.py` (for the
markdown architecture and plan docs). Each script takes a list of
files as positional arguments and operates idempotently.

The scripts use Python `re` with `\b` word boundaries to avoid catching
substrings inside other identifiers (e.g. `slave_idx` rewrites cleanly
to `nonmortar_idx`, but `slavery` — were it ever to appear — would not
be touched).
