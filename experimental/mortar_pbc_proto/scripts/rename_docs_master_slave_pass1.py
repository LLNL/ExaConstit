#!/usr/bin/env python3
"""Doc rename — handles both operational master/slave and 'master doc'."""
import os, re, sys

SUBSTITUTIONS = [
    # Doc-hierarchy uses (very specific phrases first)
    (r'\bmaster architecture doc\b',  'top-level architecture doc'),
    (r'\bthe master architecture\b',  'the top-level architecture'),
    (r'\bmaster doc\b',               'architecture doc'),
    (r'\bmaster MORTAR_PBC_ARCHITECTURE\b', 'top-level MORTAR_PBC_ARCHITECTURE'),
    (r'\b\(the "master doc"\)\b',     '(the top-level architecture doc)'),
    (r'\bMaster architecture doc\b',  'Top-level architecture doc'),
    (r'\bthe master\b(?= doc)',       'the top-level'),  # e.g. "the master doc"
    (r'\bMaster doc\b',               'Architecture doc'),

    # Operational uses (compound)
    (r'\bslave-DOF-ownership\b',      'nonmortar-DOF-ownership'),
    (r'\bslave-DOF-owner\b',          'nonmortar-DOF-owner'),
    (r'\bslave-DOF owner\b',          'nonmortar-DOF owner'),
    (r'\bslave-DOF owners\b',         'nonmortar-DOF owners'),
    (r'\bslave-DOF ownership\b',      'nonmortar-DOF ownership'),
    (r'\bslave-DOF\b',                'nonmortar-DOF'),
    (r'\bslave DOF\b',                'nonmortar DOF'),
    (r'\bslave DOFs\b',               'nonmortar DOFs'),
    (r'\bmaster-side\b',              'mortar-side'),
    (r'\bslave-side\b',               'nonmortar-side'),
    (r'\bmaster side\b',              'mortar side'),
    (r'\bslave side\b',               'nonmortar side'),
    (r'\bmaster-slave\b',             'mortar-nonmortar'),
    (r'\bslave-master\b',             'nonmortar-mortar'),
    (r'\bmaster/slave\b',             'mortar/nonmortar'),
    (r'\bslave/master\b',             'nonmortar/mortar'),
    (r'\bslave-master partners\b',    'nonmortar-mortar partners'),
    (r'\bslave-master pair\b',        'nonmortar-mortar pair'),
    (r'\bslave-master pairs\b',       'nonmortar-mortar pairs'),

    # Operational (singular)
    (r'\bmaster element\b',           'mortar element'),
    (r'\bmaster elements\b',          'mortar elements'),
    (r'\bslave element\b',            'nonmortar element'),
    (r'\bslave elements\b',           'nonmortar elements'),
    (r'\bmaster face\b',              'mortar face'),
    (r'\bmaster faces\b',             'mortar faces'),
    (r'\bslave face\b',               'nonmortar face'),
    (r'\bslave faces\b',              'nonmortar faces'),
    (r'\bmaster edge\b',              'mortar edge'),
    (r'\bmaster edges\b',             'mortar edges'),
    (r'\bslave edge\b',               'nonmortar edge'),
    (r'\bslave edges\b',              'nonmortar edges'),
    (r'\bmaster pair\b',              'mortar pair'),
    (r'\bmaster pairs\b',             'mortar pairs'),
    (r'\bslave pair\b',               'nonmortar pair'),
    (r'\bslave pairs\b',              'nonmortar pairs'),
    (r'\bmaster nodes\b',             'mortar nodes'),
    (r'\bmaster node\b',              'mortar node'),
    (r'\bslave nodes\b',              'nonmortar nodes'),
    (r'\bslave node\b',               'nonmortar node'),
    (r'\bmaster partner\b',           'mortar partner'),
    (r'\bmaster partners\b',          'mortar partners'),
    (r'\bslave rank\b',               'nonmortar rank'),
    (r'\bmaster rank\b',              'mortar rank'),
    (r'\bmaster-DOF\b',               'mortar-DOF'),
    (r'\bmaster DOF\b',               'mortar DOF'),

    # Identifier-style references in code blocks within docs
    (r'\bis_master\b', 'is_mortar'),
    (r'\bis_non_mortar\b', 'is_nonmortar'),
    (r'\b_MASTER_LABELS\b',    '_MORTAR_LABELS'),
    (r'\bmaster_node_perm\b',  'mortar_node_perm'),
    (r'\bmaster_idx\b',        'mortar_idx'),
    (r'\bslave_idx\b',         'nonmortar_idx'),
    (r'\bmaster_elems\b',      'mortar_elems'),
    (r'\bslave_elems\b',       'nonmortar_elems'),
    (r'\bmaster_face_name\b',  'mortar_face_name'),
    (r'\bslave_face_name\b',   'nonmortar_face_name'),
    (r'\bmaster_gtdofs\b',     'mortar_gtdofs'),
    (r'\bslave_gtdofs\b',      'nonmortar_gtdofs'),
    (r'\bn_master\b',          'n_mortar'),
    (r'\bn_slave\b',           'n_nonmortar'),
    (r'\bN_master_at_q\b',     'N_mortar_at_q'),
    (r'\bN_slave\b',           'N_nonmortar'),
    (r'\bN_master\b',          'N_mortar'),
    (r'\bM_slave\b',           'M_nonmortar'),
    (r'\bg_slave\b',           'g_nonmortar'),
    (r'\bg_master\b',          'g_mortar'),
    (r'\bL_master\b',          'L_mortar'),
    (r'\bL_slave\b',           'L_nonmortar'),

    # Catch-all bare words last
    (r'\bslaves\b',  'nonmortars'),
    (r'\bSlaves\b',  'Nonmortars'),
    (r'\bSLAVES\b',  'NONMORTARS'),
    (r'\bslave\b',   'nonmortar'),
    (r'\bSlave\b',   'Nonmortar'),
    (r'\bSLAVE\b',   'NONMORTAR'),
    (r'\bmasters\b', 'mortars'),
    (r'\bMasters\b', 'Mortars'),
    (r'\bMASTERS\b', 'MORTARS'),
    (r'\bmaster\b',  'mortar'),
    (r'\bMaster\b',  'Mortar'),
    (r'\bMASTER\b',  'MORTAR'),
]

COMPILED = [(re.compile(pat), repl) for pat, repl in SUBSTITUTIONS]

def migrate_file(path):
    with open(path) as fp: src = fp.read()
    new = src
    n = 0
    for pat, repl in COMPILED:
        new, k = pat.subn(repl, new)
        n += k
    if new != src:
        with open(path, 'w') as fp: fp.write(new)
    return n

if __name__ == "__main__":
    grand = 0
    for f in sys.argv[1:]:
        if not os.path.isfile(f): continue
        n = migrate_file(f)
        grand += n
        if n: print(f"  {n:5d}  {f}")
    print(f"\n  Total: {grand}")
