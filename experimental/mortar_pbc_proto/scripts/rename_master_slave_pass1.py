#!/usr/bin/env python3
"""One-shot rename: master/slave → mortar/nonmortar across the Python prototype.

Run from /home/claude/mortar_pbc_proto. Idempotent on already-migrated files.

NAMING CONVENTION applied:
  * Boolean field renames:  is_master -> is_mortar
                            is_non_mortar -> is_nonmortar
  * Operational identifiers:
      slave_*   -> nonmortar_*
      master_*  -> mortar_*
      Master*   -> Mortar*  (CamelCase / class-method names)
      Slave*    -> Nonmortar*
  * Module-level constants: _MASTER_LABELS -> _MORTAR_LABELS
                            _SLAVE_LABELS  -> _NONMORTAR_LABELS
  * Documentation prose:    "slave"/"master" -> "nonmortar"/"mortar"
  * Mathematical naming (kept unchanged):
      D^{nm} stays "D_nm" (the "nm" is the math superscript, not master/slave)
      A^m   stays "A_m"
"""
from __future__ import annotations
import os
import re
import sys

# Substitutions, applied in order. Each entry is (regex_pattern, replacement).
# Patterns use word boundaries (`\b`) to avoid matching substrings inside
# other identifiers.
SUBSTITUTIONS: list[tuple[str, str]] = [
    # ---- Module-level constants (must come before generic master/slave) ----
    (r'\b_MASTER_LABELS\b',    '_MORTAR_LABELS'),
    (r'\b_SLAVE_LABELS\b',     '_NONMORTAR_LABELS'),

    # ---- CamelCase class / function names ----
    (r'\bMortarFaceAssembler\b',          'MortarFaceAssembler'),  # no change (the class is correctly named)
    (r'\bMasterFaceAssembler\b',          'MortarFaceAssembler'),  # if any old name remains
    # (Other CamelCase aren't currently in the codebase; skip.)

    # ---- Method-name fragments (snake_case) ----
    (r'\b_master_node_permutation_apply\b', '_mortar_node_permutation_apply'),
    (r'\b_eval_slave_dual\b',               '_eval_nonmortar_dual'),
    (r'\b_eval_slave_shape\b',              '_eval_nonmortar_shape'),
    (r'\b_eval_master_shape\b',             '_eval_mortar_shape'),
    (r'\b_slave_jacobian\b',                '_nonmortar_jacobian'),
    (r'\b_reorder_master_shape\b',          '_reorder_mortar_shape'),
    (r'\bmatch_conforming_face_pairs\b',    'match_conforming_face_pairs'),  # no change

    # ---- Common identifiers ----
    # Boolean field renames (must come BEFORE generic 'master'/'slave' rules
    # because is_master matches the bare 'master' rule otherwise).
    (r'\bis_non_mortar\b', 'is_nonmortar'),
    (r'\bis_master\b',     'is_mortar'),

    # Pair-match indices and permutations
    (r'\bmaster_node_perm\b',  'mortar_node_perm'),
    (r'\bmaster_idx_match\b',  'mortar_idx_match'),
    (r'\bmaster_idx\b',        'mortar_idx'),
    (r'\bslave_idx\b',         'nonmortar_idx'),

    # Element / geometry args
    (r'\bslave_elems\b',     'nonmortar_elems'),
    (r'\bmaster_elems\b',    'mortar_elems'),
    (r'\bslave_elem\b',      'nonmortar_elem'),
    (r'\bmaster_elem\b',     'mortar_elem'),
    (r'\bmaster_centroids\b','mortar_centroids'),
    (r'\bmaster_centroid\b', 'mortar_centroid'),
    (r'\bs_centroid_3d\b',   's_centroid_3d'),    # no change
    (r'\bs_centroid_inplane\b', 's_centroid_inplane'),  # no change

    # Names / strings
    (r'\bslave_face_name\b',  'nonmortar_face_name'),
    (r'\bmaster_face_name\b', 'mortar_face_name'),
    (r'\bslave_name\b',       'nonmortar_name'),
    (r'\bmaster_name\b',      'mortar_name'),
    (r'\bslave_face\b',       'nonmortar_face'),
    (r'\bmaster_face\b',      'mortar_face'),
    (r'\bslave_edge\b',       'nonmortar_edge'),
    (r'\bmaster_edge\b',      'mortar_edge'),

    # GTDof maps
    (r'\bslave_gtdofs\b',  'nonmortar_gtdofs'),
    (r'\bmaster_gtdofs\b', 'mortar_gtdofs'),
    (r'\bslave_row_of\b',  'nonmortar_row_of'),
    (r'\bmaster_col_of\b', 'mortar_col_of'),
    (r'\bn_master\b',      'n_mortar'),
    (r'\bn_slave\b',       'n_nonmortar'),

    # Locals in matching helpers
    (r'\bslave_local\b',  'nonmortar_local'),
    (r'\bmaster_local\b', 'mortar_local'),

    # Quadrature / shape evaluation
    (r'\bM_slave\b',  'M_nonmortar'),
    (r'\bN_slave\b',  'N_nonmortar'),
    (r'\bN_master\b', 'N_mortar'),
    (r'\bN_master_in_master_local\b', 'N_mortar_in_mortar_local'),  # safety
    (r'\bq_pt_slave\b',  'q_pt_nonmortar'),
    (r'\bq_pt_master\b', 'q_pt_mortar'),
    (r'\bxi_on_slave\b',  'xi_on_nonmortar'),  # if appears
    (r'\bxi_on_master\b', 'xi_on_mortar'),     # if appears

    # Coordinate-related
    (r'\bs_coords_in\b',     's_coords_in'),    # no change
    (r'\bm_coords_in\b',     'm_coords_in'),    # no change
    (r'\bslave_coords\b',    'nonmortar_coords'),
    (r'\bmaster_coords\b',   'mortar_coords'),

    # MasterRef / MasterBary helpers (used in some places)
    (r'\bmaster_at_slave_0\b', 'mortar_at_nonmortar_0'),
    (r'\bmaster_at_slave_1\b', 'mortar_at_nonmortar_1'),
    (r'\bmaster_at_slave_2\b', 'mortar_at_nonmortar_2'),
    (r'\bmaster_at_slave_3\b', 'mortar_at_nonmortar_3'),
    (r'\bmaster_q_pt\b',       'mortar_q_pt'),

    # ---- Hyphenated forms in prose / comments ----
    (r'\bslave-side\b',  'nonmortar-side'),
    (r'\bmaster-side\b', 'mortar-side'),
    (r'\bslave-master\b', 'nonmortar-mortar'),
    (r'\bmaster-slave\b', 'mortar-nonmortar'),

    # ---- Bare words (last; they catch documentation prose) ----
    (r'\bslave\b',   'nonmortar'),
    (r'\bSlave\b',   'Nonmortar'),
    (r'\bSLAVE\b',   'NONMORTAR'),
    (r'\bslaves\b',  'nonmortars'),     # might be matched by \bslave\b first; keep for safety
    (r'\bMASTER\b',  'MORTAR'),
    (r'\bMaster\b',  'Mortar'),
    (r'\bmaster\b',  'mortar'),
    (r'\bmasters\b', 'mortars'),
]

# Compile all patterns once.
COMPILED = [(re.compile(pat), repl) for pat, repl in SUBSTITUTIONS]


def migrate_file(path: str) -> tuple[int, int]:
    """Apply all substitutions to a file. Returns (lines_changed, total_substitutions)."""
    with open(path, 'r', encoding='utf-8') as fp:
        original = fp.read()
    new = original
    total_subs = 0
    for pat, repl in COMPILED:
        new, n = pat.subn(repl, new)
        total_subs += n
    if new != original:
        with open(path, 'w', encoding='utf-8') as fp:
            fp.write(new)
    # Count changed lines (rough proxy)
    orig_lines = original.splitlines()
    new_lines = new.splitlines()
    diff_count = sum(1 for o, n in zip(orig_lines, new_lines) if o != n)
    diff_count += abs(len(orig_lines) - len(new_lines))
    return diff_count, total_subs


def main() -> int:
    targets = sys.argv[1:]
    if not targets:
        print("usage: rename_master_slave.py <file1> [<file2> ...]")
        return 1
    grand_total = 0
    for path in targets:
        if not os.path.isfile(path):
            print(f"  SKIP   {path} (not a regular file)")
            continue
        lines, subs = migrate_file(path)
        grand_total += subs
        print(f"  {subs:5d} subs / {lines:5d} lines changed   {path}")
    print(f"\n  Total substitutions: {grand_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
