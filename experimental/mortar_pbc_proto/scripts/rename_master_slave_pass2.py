#!/usr/bin/env python3
"""Second-pass rename for missed identifiers."""
from __future__ import annotations
import os, re, sys

SUBSTITUTIONS = [
    # Multi-component matches first (longer patterns)
    (r'\bslave_quads_master_tris\b',  'nonmortar_quads_mortar_tris'),
    (r'\bslave_tris_master_quads\b',  'nonmortar_tris_mortar_quads'),
    (r'\btest_match_conforming_face_pairs_shuffled_master_order\b',
     'test_match_conforming_face_pairs_shuffled_mortar_order'),

    # Compound identifiers
    (r'\bn_master_kept\b',                'n_mortar_kept'),
    (r'\bn_slave_kept\b',                 'n_nonmortar_kept'),
    (r'\bok_masters\b',                   'ok_mortars'),
    (r'\bn_master_faces\b',               'n_mortar_faces'),
    (r'\bn_master_edges\b',               'n_mortar_edges'),
    (r'\bg_slave\b',                      'g_nonmortar'),
    (r'\bg_master\b',                     'g_mortar'),
    (r'\bN_master_at_q\b',                'N_mortar_at_q'),
    (r'\bL_master\b',                     'L_mortar'),
    (r'\bL_slave\b',                      'L_nonmortar'),
    (r'\bboth_slaves\b',                  'both_nonmortars'),
    (r'\bu_slave_c\b',                    'u_nonmortar_c'),
    (r'\bu_master_c\b',                   'u_mortar_c'),
    (r'\bn_kept_slave_face_dofs\b',       'n_kept_nonmortar_face_dofs'),
    (r'\bn_interior_slave_nodes\b',       'n_interior_nonmortar_nodes'),
    (r'\bmaster_X\b',                     'mortar_X'),
    (r'\bslave_X\b',                      'nonmortar_X'),
    (r'\bmaster_by_axis\b',               'mortar_by_axis'),
    (r'\bslaves_by_axis\b',               'nonmortars_by_axis'),
    (r'\bmaster_g_xyz\b',                 'mortar_g_xyz'),
    (r'\bslave_g_xyz\b',                  'nonmortar_g_xyz'),
    (r'\bmaster_gtdofs_kept\b',           'mortar_gtdofs_kept'),
    (r'\bslave_gtdofs_kept\b',            'nonmortar_gtdofs_kept'),
    (r'\bmaster_gx\b',                    'mortar_gx'),
    (r'\bslave_gx\b',                     'nonmortar_gx'),
    (r'\bmaster_has_both\b',              'mortar_has_both'),
    (r'\bslave_has_both\b',               'nonmortar_has_both'),
    (r'\bmaster_l\b',                     'mortar_l'),
    (r'\bslave_k\b',                      'nonmortar_k'),
    (r'\bmaster_label\b',                 'mortar_label'),
    (r'\bslave_label\b',                  'nonmortar_label'),
    (r'\bmaster_perp_coords\b',           'mortar_perp_coords'),
    (r'\bslave_perp\b',                   'nonmortar_perp'),
    (r'\bmaster_q\b',                     'mortar_q'),
    (r'\bslave_q\b',                      'nonmortar_q'),
    (r'\bslave_q_pt\b',                   'nonmortar_q_pt'),
    (r'\bmaster_quads\b',                 'mortar_quads'),
    (r'\bslave_quads\b',                  'nonmortar_quads'),
    (r'\bmaster_shuffled\b',              'mortar_shuffled'),
    (r'\bmaster_t\b',                     'mortar_t'),
    (r'\bslave_t\b',                      'nonmortar_t'),
    (r'\bmaster_tdof\b',                  'mortar_tdof'),
    (r'\bslave_tdof\b',                   'nonmortar_tdof'),
    (r'\bmaster_tris\b',                  'mortar_tris'),
    (r'\bslave_tris\b',                   'nonmortar_tris'),
    (r'\bslave_J_fn\b',                   'nonmortar_J_fn'),
    (r'\bslave_mod\b',                    'nonmortar_mod'),
    (r'\bslave_unmod\b',                  'nonmortar_unmod'),
]

COMPILED = [(re.compile(pat), repl) for pat, repl in SUBSTITUTIONS]

def migrate_file(path):
    with open(path) as fp: src = fp.read()
    new = src
    n_total = 0
    for pat, repl in COMPILED:
        new, n = pat.subn(repl, new)
        n_total += n
    if new != src:
        with open(path, 'w') as fp: fp.write(new)
    return n_total

if __name__ == "__main__":
    grand = 0
    for f in sys.argv[1:]:
        if not os.path.isfile(f): continue
        n = migrate_file(f)
        grand += n
        if n: print(f"  {n:5d}  {f}")
    print(f"\n  Total: {grand}")
