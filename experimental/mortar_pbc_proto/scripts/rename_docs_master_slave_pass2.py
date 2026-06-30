#!/usr/bin/env python3
"""Final pass for doc residuals."""
import os, re, sys

SUBS = [
    # Compound identifiers in pseudocode blocks
    (r'\bn_master_kept\b',            'n_mortar_kept'),
    (r'\bn_slave_kept\b',             'n_nonmortar_kept'),
    (r'\bN_master_at_m\b',            'N_mortar_at_m'),
    (r'\bN_dropped_master\b',         'N_dropped_mortar'),
    (r'\b_eval_master_shape\b',       '_eval_mortar_shape'),
    (r'\b_eval_slave_dual\b',         '_eval_nonmortar_dual'),
    (r'\b_eval_slave_shape\b',        '_eval_nonmortar_shape'),
    (r'\b_slave_jacobian\b',          '_nonmortar_jacobian'),
    (r'\bcorner_master\b',            'corner_mortar'),
    (r'\blocate_master\b',            'locate_mortar'),
    (r'\bmaster_face_axis\b',         'mortar_face_axis'),
    (r'\bmaster_face\b',              'mortar_face'),
    (r'\bslave_face\b',               'nonmortar_face'),
    (r'\bmaster_edge\b',              'mortar_edge'),
    (r'\bslave_edge\b',               'nonmortar_edge'),
    (r'\bmaster_edges\b',             'mortar_edges'),
    (r'\bslave_edges\b',              'nonmortar_edges'),
    (r'\bmaster_quad_id\b',           'mortar_quad_id'),
    (r'\bmaster_tri_id\b',            'mortar_tri_id'),
    (r'\bmaster_line_id\b',           'mortar_line_id'),
    (r'\bmaster_elem\b',              'mortar_elem'),
    (r'\bmaster_quads\b',             'mortar_quads'),
    (r'\bslave_quads\b',              'nonmortar_quads'),
    (r'\bmaster_tris\b',              'mortar_tris'),
    (r'\bslave_tris\b',               'nonmortar_tris'),
    (r'\bslave_LM_DOFs\b',            'nonmortar_LM_DOFs'),
    (r'\bslave_DOFs\b',               'nonmortar_DOFs'),
    (r'\bmaster_DOFs\b',              'mortar_DOFs'),
    (r'\bu_master\b',                 'u_mortar'),
    (r'\bu_slave\b',                  'u_nonmortar'),
    (r'\bx_master\b',                 'x_mortar'),
    (r'\bx_slave\b',                  'x_nonmortar'),
    (r'\bslave_gtdofs_per_component\b', 'nonmortar_gtdofs_per_component'),
    (r'\bmaster_gtdofs_per_component\b','mortar_gtdofs_per_component'),

    # Unicode pseudocode (xi/eta/lambda)
    (r'ξ_master', 'ξ_mortar'),
    (r'ξ_slave',  'ξ_nonmortar'),
    (r'η_master', 'η_mortar'),
    (r'η_slave',  'η_nonmortar'),
    (r'λ_master', 'λ_mortar'),
    (r'λ_slave',  'λ_nonmortar'),

    # The prefix `_slave` (when not part of a longer identifier)
    # This handles things like `S in _slave_face` -> `S in _nonmortar_face`
    # but careful — should be caught by other rules already

    # Final catch-all for plain words. These only fire for things the
    # word-boundary regex above missed.
    (r'\bmasters\b',  'mortars'),
    (r'\bslaves\b',   'nonmortars'),
    (r'\bmaster\b',   'mortar'),
    (r'\bslave\b',    'nonmortar'),
    (r'\bMaster\b',   'Mortar'),
    (r'\bSlave\b',    'Nonmortar'),
    (r'\bMASTER\b',   'MORTAR'),
    (r'\bSLAVE\b',    'NONMORTAR'),
]
COMPILED = [(re.compile(p), r) for p, r in SUBS]

def main():
    grand = 0
    for f in sys.argv[1:]:
        if not os.path.isfile(f): continue
        with open(f) as fp: src = fp.read()
        new = src
        n = 0
        for pat, repl in COMPILED:
            new, k = pat.subn(repl, new)
            n += k
        if new != src:
            with open(f, 'w') as fp: fp.write(new)
        grand += n
        if n: print(f"  {n:5d}  {f}")
    print(f"\n  Total: {grand}")

if __name__ == "__main__":
    main()
