#!/usr/bin/env python3
"""
Migration script for ExaConstit option files from old format to new format.
Converts old option_parser.cpp/hpp TOML format to option_parser_v2.cpp/hpp format.

Key features:
- Handles case-insensitive option values from old parser
- Normalizes solver names (e.g., PCG -> CG)
- Converts various spelling variations to canonical v2 format
- Normalizes boolean values (true/True/TRUE/yes/1 -> true)
- Preserves backward compatibility while enforcing v2 conventions

The script automatically handles common variations in:
- Solver types (PCG->CG, various cases)
- Assembly modes (pa/PA, full/FULL, etc.)
- Material models (ExaCMech/exacmech/EXACMECH)
- Crystal types (fcc/FCC/Fcc)
- Boolean values (true/True/yes/1/on)
- And many more...
"""

import toml
from toml.encoder import TomlEncoder
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import argparse
from collections import OrderedDict
import copy

from tomlkit import document, table, inline_table, dumps
from collections.abc import Mapping, Sequence

def dict_to_toml(data: Mapping) -> document:
    """
    Recursively convert a nested dict/list structure into a tomlkit.document,
    using:
      - table()      for any dict ⇒ creates [section] or [a.b.c]
      - inline_table() for dicts inside lists ⇒ { key = value, ... }
      - plain lists  for list of scalars or inline tables
    """
    def _populate(container, obj):
        for key, val in obj.items():
            # 1) sub‐dict ⇒ new table (always emits a [parent.child] block)
            if isinstance(val, Mapping):
                sub_tbl = table()
                _populate(sub_tbl, val)
                container.add(key, sub_tbl)

            # 2) list ⇒ may need inline tables for dict‐elements
            elif isinstance(val, Sequence) and not isinstance(val, str):
                new_list = []
                for item in val:
                    if isinstance(item, Mapping):
                        # inline dicts in lists become { … } entries
                        itbl = inline_table()
                        _populate(itbl, item)
                        new_list.append(itbl)
                    else:
                        new_list.append(item)
                container.add(key, new_list)

            # 3) scalar ⇒ direct add
            else:
                container.add(key, val)

    doc = document()
    _populate(doc, data)
    return doc

def dumps_indented(data, indent_str='    '):
    """
    Dump `data` to TOML, indenting each table's keys by `indent_str`.
    """
    doc = dict_to_toml(data)
    raw = dumps(doc)
    lines = raw.splitlines(keepends=True)
    out_lines = []
    current_indent = ''

    for line in lines:
        # if it's a table header, reset/increase indent
        stripped = line.lstrip()
        sub_field_cnt = stripped.count('.')
        if stripped.startswith('[[') or stripped.startswith('['):
            ind_indent = ''
            ind_indent = indent_str * sub_field_cnt
            current_indent = indent_str * (sub_field_cnt + 1)
            out_lines.append(ind_indent + line)
        # blank lines pass through
        elif stripped.strip() == '':
            out_lines.append(line)
        else:
            # indent key/value lines
            out_lines.append(current_indent + line)

    return ''.join(out_lines)

class OptionMigrator:
    """Migrates old ExaConstit option files to the new format."""
    
    def __init__(self):
        self.old_config = {}
        self.new_config = OrderedDict({})
        self.warnings = []
        
    def load_old_config(self, filepath: str) -> Dict[str, Any]:
        """Load the old format TOML file."""
        with open(filepath, 'r') as f:
            self.old_config = toml.load(f)
        return self.old_config
    
    def migrate(self) -> Dict[str, Any]:
        """Main migration function."""
        self.new_config = OrderedDict({})
        
        # Migrate version info
        if 'Version' in self.old_config:
            self.new_config['Version'] = self.old_config['Version']
        
        # Migrate basename
        if 'basename' in self.old_config:
            self.new_config['basename'] = self.old_config['basename']

        # Migrate materials and properties
        self._migrate_materials()

        # Migrate boundary conditions
        self._migrate_bcs()

        # Migrate time options
        self._migrate_time()

        # Migrate solver options
        self._migrate_solvers()

        # Migrate visualization options
        self._migrate_visualizations()

        # Migrate post-processing options (new in v2)
        self._migrate_post_processing()

        # Migrate mesh options
        self._migrate_mesh()

        return self.new_config
    
    def _migrate_mesh(self):
        """Migrate mesh configuration."""
        if 'Mesh' not in self.old_config:
            return
            
        old_mesh = self.old_config['Mesh']
        new_mesh = {}
        
        # Mesh type mapping
        if 'type' in old_mesh:
            new_mesh['type'] = self._normalize_string_option(
                old_mesh['type'], 'mesh_type'
            )
        elif 'mesh_type' in old_mesh:
            new_mesh['type'] = self._normalize_string_option(
                old_mesh['mesh_type'], 'mesh_type'
            )
            
        # File location mapping
        if 'floc' in old_mesh:
            new_mesh['floc'] = old_mesh['floc']
        elif 'file' in old_mesh:
            new_mesh['floc'] = old_mesh['file']
            
        # Refinement levels - keep original names
        if 'ref_ser' in old_mesh:
            new_mesh['ref_ser'] = old_mesh['ref_ser']
        elif 'refine_serial' in old_mesh:
            new_mesh['refine_serial'] = old_mesh['refine_serial']
            
        if 'ref_par' in old_mesh:
            new_mesh['ref_par'] = old_mesh['ref_par']
        elif 'refine_parallel' in old_mesh:
            new_mesh['refine_parallel'] = old_mesh['refine_parallel']
        
        # Order/p_refinement
        if 'p_refinement' in old_mesh:
            new_mesh['p_refinement'] = old_mesh['p_refinement']
        elif 'order' in old_mesh:
            new_mesh['order'] = old_mesh['order']
            
        # Periodicity (new in v2)
        if 'periodicity' in old_mesh:
            new_mesh['periodicity'] = self._normalize_bool(old_mesh['periodicity'])
            
        # Auto mesh parameters
        if 'Auto' in old_mesh:
            auto = old_mesh['Auto']
            new_mesh['Auto'] = {}
            
            # Length/mxyz mapping
            if 'length' in auto:
                new_mesh['Auto']['mxyz'] = auto['length']
            elif 'mxyz' in auto:
                new_mesh['Auto']['mxyz'] = auto['mxyz']
                
            # ncuts/nxyz mapping
            if 'ncuts' in auto:
                new_mesh['Auto']['nxyz'] = auto['ncuts']
            elif 'nxyz' in auto:
                new_mesh['Auto']['nxyz'] = auto['nxyz']
            
        self.new_config['Mesh'] = new_mesh
    
    def _migrate_time(self):
        """Migrate time stepping configuration."""
        if 'Time' not in self.old_config:
            return
            
        old_time = self.old_config['Time']
        new_time = {}
        
        # Check for nested time configurations first (highest priority)
        if 'Custom' in old_time:
            # Custom time stepping takes highest priority
            new_time['time_type'] = 'custom'
            new_time['Custom'] = {
                'floc': old_time['Custom'].get('floc', 'custom_dt.txt'),
                'nsteps': old_time['Custom'].get('nsteps', 1)
            }
        elif 'Auto' in old_time:
            # Auto time stepping is second priority
            new_time['time_type'] = 'auto'
            auto_config = old_time['Auto']
            new_time['Auto'] = {
                'dt_start': auto_config.get('dt_start', auto_config.get('dt', 1.0)),
                'dt_min': auto_config.get('dt_min', 0.001),
                'dt_max': auto_config.get('dt_max', 1000.0),
                'dt_scale': auto_config.get('dt_scale', 0.25),
                't_final': auto_config.get('t_final', 1.0)
            }
            if 'auto_dt_file' in auto_config:
                new_time['Auto']['auto_dt_file'] = auto_config['auto_dt_file']
        elif 'Fixed' in old_time:
            # Fixed time stepping is third priority
            new_time['time_type'] = 'fixed'
            fixed_config = old_time['Fixed']
            new_time['Fixed'] = {
                'dt': fixed_config.get('dt', 1.0),
                't_final': fixed_config.get('t_final', 1.0)
            }
        else:
            # Legacy format - check for dt_cust/dt_auto flags
            if 'dt_file' in old_time and self._normalize_bool(old_time.get('dt_cust', False)):
                # Custom time stepping
                new_time['time_type'] = 'custom'
                new_time['Custom'] = {
                    'floc': old_time['dt_file'],
                    'nsteps': old_time.get('nsteps', 1)
                }
            elif self._normalize_bool(old_time.get('dt_auto', False)):
                # Auto time stepping
                new_time['time_type'] = 'auto'
                new_time['Auto'] = {
                    'dt_start': old_time.get('dt', 1.0),
                    'dt_min': old_time.get('dt_min', 0.001),
                    'dt_max': old_time.get('dt_max', 1000.0),
                    'dt_scale': old_time.get('dt_scale', 0.25),
                    't_final': old_time.get('t_final', 1.0)
                }
                if 'auto_dt_file' in old_time:
                    new_time['Auto']['auto_dt_file'] = old_time['auto_dt_file']
            else:
                # Fixed time stepping (default)
                new_time['time_type'] = 'fixed'
                new_time['Fixed'] = {
                    'dt': old_time.get('dt', 1.0),
                    't_final': old_time.get('t_final', 1.0)
                }
        
        # Normalize time type
        new_time['time_type'] = self._normalize_string_option(
            new_time['time_type'], 'time_type'
        )
        
        # Restart options (if present)
        if 'restart' in old_time:
            new_time['restart'] = self._normalize_bool(old_time['restart'])
        if 'restart_time' in old_time:
            new_time['restart_time'] = old_time['restart_time']
        if 'restart_cycle' in old_time:
            new_time['restart_cycle'] = old_time['restart_cycle']
            
        self.new_config['Time'] = new_time
    
    def _normalize_bool(self, value: Any) -> bool:
        """Normalize boolean values from various representations."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 't', 'yes', 'y', '1', 'on')
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    def _is_nested_array(self, data: Any) -> bool:
        """Check if data is a nested array (list of lists)"""
        return (isinstance(data, list) and 
                len(data) > 0 and 
                isinstance(data[0], list))
    
    def _normalize_string_option(self, value: str, option_type: str) -> str:
        """Normalize string options to match v2 parser expectations."""
        value = value.strip()
        
        # Linear solver type normalization
        linear_solver_map = {
            'pcg': 'CG',
            'PCG': 'CG',
            'cg': 'CG',
            'CG': 'CG',
            'gmres': 'GMRES',
            'GMRES': 'GMRES',
            'minres': 'MINRES',
            'MINRES': 'MINRES'
        }
        
        # Preconditioner normalization
        preconditioner_map = {
            'jacobi': 'JACOBI',
            'Jacobi': 'JACOBI',
            'JACOBI': 'JACOBI',
            'amg': 'AMG',
            'Amg': 'AMG',
            'AMG': 'AMG'
        }
        
        # Assembly type normalization
        assembly_map = {
            'full': 'FULL',
            'Full': 'FULL',
            'FULL': 'FULL',
            'pa': 'PA',
            'Pa': 'PA',
            'PA': 'PA',
            'ea': 'EA',
            'Ea': 'EA',
            'EA': 'EA'
        }
        
        # RT Model normalization
        rtmodel_map = {
            'cpu': 'CPU',
            'Cpu': 'CPU',
            'CPU': 'CPU',
            'openmp': 'OPENMP',
            'OpenMP': 'OPENMP',
            'OPENMP': 'OPENMP',
            'gpu': 'GPU',
            'Gpu': 'GPU',
            'GPU': 'GPU'
        }
        
        # Nonlinear solver normalization
        nl_solver_map = {
            'nr': 'NR',
            'Nr': 'NR',
            'NR': 'NR',
            'newton': 'NR',
            'Newton': 'NR',
            'NEWTON': 'NR',
            'nrls': 'NRLS',
            'NRLS': 'NRLS',
            'Nrls': 'NRLS'
        }
        
        # Material model type normalization
        mech_type_map = {
            'umat': 'umat',
            'Umat': 'umat',
            'UMAT': 'umat',
            'exacmech': 'exacmech',
            'ExaCMech': 'exacmech',
            'EXACMECH': 'exacmech',
            'exaconstit': 'exacmech',  # Common typo/variation
            'ExaConstit': 'exacmech'
        }
        
        # Crystal type normalization (for ExaCMech)
        xtal_type_map = {
            'fcc': 'FCC',
            'Fcc': 'FCC',
            'FCC': 'FCC',
            'bcc': 'BCC',
            'Bcc': 'BCC',
            'BCC': 'BCC',
            'hcp': 'HCP',
            'Hcp': 'HCP',
            'HCP': 'HCP'
        }
        
        # Slip type normalization (for ExaCMech)
        slip_type_map = {
            'powervoce': 'PowerVoce',
            'PowerVoce': 'PowerVoce',
            'POWERVOCE': 'PowerVoce',
            'power_voce': 'PowerVoce',
            'powervocenl': 'PowerVoceNL',
            'PowerVoceNL': 'PowerVoceNL',
            'POWERVOCENL': 'PowerVoceNL',
            'power_voce_nl': 'PowerVoceNL',
            'mtsdd': 'MTSDD',
            'Mtsdd': 'MTSDD',
            'MTSDD': 'MTSDD'
        }
        
        # Integration model normalization
        integ_model_map = {
            'default': 'DEFAULT',
            'Default': 'DEFAULT',
            'DEFAULT': 'DEFAULT',
            'bbar': 'BBAR',
            'Bbar': 'BBAR',
            'BBAR': 'BBAR',
            'b-bar': 'BBAR',
            'B-bar': 'BBAR'
        }
        
        # Time type normalization
        time_type_map = {
            'fixed': 'fixed',
            'Fixed': 'fixed',
            'FIXED': 'fixed',
            'auto': 'auto',
            'Auto': 'auto',
            'AUTO': 'auto',
            'adaptive': 'auto',  # Common alternative
            'custom': 'custom',
            'Custom': 'custom',
            'CUSTOM': 'custom'
        }
        
        # Mesh type normalization
        mesh_type_map = {
            'file': 'file',
            'File': 'file',
            'FILE': 'file',
            'auto': 'auto',
            'Auto': 'auto',
            'AUTO': 'auto'
        }
        
        # Orientation type normalization
        ori_type_map = {
            'euler': 'euler',
            'Euler': 'euler', 
            'EULER': 'euler',
            'quat': 'quat',
            'Quat': 'quat',
            'QUAT': 'quat',
            'quaternion': 'quat',
            'Quaternion': 'quat',
            'QUATERNION': 'quat',
            'custom': 'custom',
            'Custom': 'custom',
            'CUSTOM': 'custom'
        }
        
        # Grain field name normalization
        grain_field_map = {
            'ori_floc': 'orientation_file',
            'grain_floc': 'grain_file'
        }
        
        # Select appropriate map based on option type
        normalization_maps = {
            'linear_solver': linear_solver_map,
            'preconditioner': preconditioner_map,
            'assembly': assembly_map,
            'rtmodel': rtmodel_map,
            'nl_solver': nl_solver_map,
            'mech_type': mech_type_map,
            'xtal_type': xtal_type_map,
            'slip_type': slip_type_map,
            'integ_model': integ_model_map,
            'time_type': time_type_map,
            'mesh_type': mesh_type_map,
            'ori_type': ori_type_map,
            'grain_field': grain_field_map
        }
        
        if option_type in normalization_maps:
            mapping = normalization_maps[option_type]
            if value in mapping:
                normalized = mapping[value]
                if value != normalized:
                    self.warnings.append(f"Normalized '{value}' to '{normalized}' for {option_type}")
                return normalized
            else:
                self.warnings.append(f"Unknown {option_type} value: '{value}' - using as-is")
                return value
        
        return value

    def _migrate_solvers(self):
        """Migrate solver configuration."""
        if 'Solvers' not in self.old_config:
            return
            
        old_solvers = self.old_config['Solvers']
        new_solvers = {}
        
        # Assembly options
        if 'assembly' in old_solvers:
            new_solvers['assembly'] = self._normalize_string_option(
                old_solvers['assembly'], 'assembly'
            )
        if 'rtmodel' in old_solvers:
            new_solvers['rtmodel'] = self._normalize_string_option(
                old_solvers['rtmodel'], 'rtmodel'
            )
        if 'integ_model' in old_solvers:
            new_solvers['integ_model'] = self._normalize_string_option(
                old_solvers['integ_model'], 'integ_model'
            )
            
        # Linear solver (Krylov)
        if 'Krylov' in old_solvers:
            new_krylov = dict(old_solvers['Krylov'])
            
            # Normalize solver type
            if 'solver' in new_krylov:
                new_krylov['solver'] = self._normalize_string_option(
                    new_krylov['solver'], 'linear_solver'
                )
            
            # Normalize preconditioner
            if 'preconditioner' in new_krylov:
                new_krylov['preconditioner'] = self._normalize_string_option(
                    new_krylov['preconditioner'], 'preconditioner'
                )
                
            new_solvers['Krylov'] = new_krylov
            
        # Nonlinear solver (Newton-Raphson)
        if 'NR' in old_solvers:
            new_nr = dict(old_solvers['NR'])
            
            # Add nl_solver type if not present
            if 'nl_solver' not in new_nr:
                new_nr['nl_solver'] = 'NR'
            else:
                new_nr['nl_solver'] = self._normalize_string_option(
                    new_nr['nl_solver'], 'nl_solver'
                )
                
            new_solvers['NR'] = new_nr
                
        self.new_config['Solvers'] = new_solvers
    
    def _migrate_materials(self):
        """Migrate material properties and model configuration."""
        materials = []
        
        # Create material from Properties and Model sections
        material = {
            'material_name': 'default',
            'region_id': 0
        }
        
        # Get temperature from Properties
        if 'Properties' in self.old_config:
            props = self.old_config['Properties']
            if 'temperature' in props:
                material['temperature'] = float(props['temperature'])
                
            # Material properties
            if 'Matl_Props' in props:
                material['Properties'] = dict(props['Matl_Props'])
            elif 'Properties' in props:  # Sometimes nested
                material['Properties'] = dict(props['Properties'])
                
            # State variables
            if 'State_Vars' in props:
                material['State_Vars'] = dict(props['State_Vars'])
                
            # Grain information
            if 'Grain' in props:
                grain = dict(props['Grain'])
                # Normalize orientation type if present
                if 'ori_type' in grain:
                    grain['ori_type'] = self._normalize_string_option(
                        grain['ori_type'], 'ori_type'
                    )
                    
                                # Normalize orientation type if present
                if 'orientation_file' in grain:
                    self.new_config['orientation_file'] = grain['orientation_file']
                if 'ori_floc' in grain:
                    grain['orientation_file'] = self._normalize_string_option(
                        grain['ori_floc'], 'ori_floc'
                    )
                    self.new_config['orientation_file'] = grain['orientation_file']
                    del grain["ori_floc"]

                if 'grain_file' in grain:
                    self.new_config['grain_file'] = grain['grain_file']
                if 'grain_floc' in grain:
                    grain['grain_file'] = self._normalize_string_option(
                        grain['grain_floc'], 'grain_floc'
                    )
                    self.new_config['grain_file'] = grain['grain_file']
                    del grain["grain_floc"]
                print(grain)
                print(self.new_config)
                material['Grain'] = grain

        # Get model configuration
        if 'Model' in self.old_config:
            model = self.old_config['Model']
            material['Model'] = {}

            if 'mech_type' in model:
                normalized_mech = self._normalize_string_option(
                    model['mech_type'], 'mech_type'
                )
                material['mech_type'] = normalized_mech
                material['Model']['mech_type'] = normalized_mech
                
            if 'cp' in model:
                material['Model']['cp'] = self._normalize_bool(model['cp'])
                
            # Model-specific options
            if 'ExaCMech' in model:
                exacmech = dict(model['ExaCMech'])
                
                # Normalize xtal_type and slip_type if present
                if 'xtal_type' in exacmech:
                    exacmech['xtal_type'] = self._normalize_string_option(
                        exacmech['xtal_type'], 'xtal_type'
                    )
                if 'slip_type' in exacmech:
                    exacmech['slip_type'] = self._normalize_string_option(
                        exacmech['slip_type'], 'slip_type'
                    )
                    
                material['Model']['ExaCMech'] = exacmech

                def getEffectiveShortCut(xtal_type, slip_type):
                    derived_shortcut = "evptn_" + xtal_type
                    #Map slip_type to the appropriate suffix
                    if (xtal_type == "FCC" or xtal_type == "BCC"):
                        if (slip_type == "PowerVoce"):
                            derived_shortcut += "_A"
                        elif (slip_type == "PowerVoceNL"):
                            derived_shortcut += "_AH"
                        elif (slip_type == "MTSDD"):
                            derived_shortcut += "_B"
                    elif (xtal_type == "HCP"):
                        if (slip_type == "MTSDD"):
                            derived_shortcut += "_A"
                    return derived_shortcut
                
                # Suggest using shortcut if xtal_type and slip_type are present
                if 'xtal_type' in exacmech and 'slip_type' in exacmech:
                    suggested_shortcut = getEffectiveShortCut(exacmech['xtal_type'], exacmech['slip_type'])
                    self.warnings.append(
                        f"ExaCMech model uses xtal_type/slip_type. "
                        f"Consider using 'shortcut = \"{suggested_shortcut}\"' instead for v2."
                    )
                    
            elif 'UMAT' in model:
                umat = dict(model['UMAT'])
                # Normalize thermal boolean if present
                if 'thermal' in umat:
                    umat['thermal'] = self._normalize_bool(umat['thermal'])
                material['Model']['UMAT'] = umat
        
        materials.append(material)

        self.new_config['Materials'] = materials
    
    def _migrate_bcs(self):
        """Migrate boundary conditions."""
        if 'BCs' not in self.old_config:
            return
            
        old_bcs = self.old_config['BCs']
        new_bcs = {}

        changing_bcs = False
        
        # Check for changing BCs
        if self._normalize_bool(old_bcs.get('changing_ess_bcs', False)):
            changing_bcs = True
            new_bcs['time_info'] = { "cycle_dependent" : True, "cycles" : old_bcs['update_steps'] }
        else:
            new_bcs['time_info'] = { "cycle_dependent" : True, "cycles" : [1] }

        def create_bc_objects_for_step(step_ids: List[int], step_comps: List[int],
                                   step_vals: List[float], step_vgrads: List[List[float]],
                                   vgrad_origin: List[float]) -> tuple:
            """
            Create velocity and velocity gradient BC objects for a single step
            
            Returns: (velocity_bc_dict or None, vgrad_bc_dict or None)
            """
            vel_ids, vel_comps = [], []
            vgrad_ids, vgrad_comps = [], []

            # Separate velocity and velocity gradient BCs
            val_idx = 0
            for i, (node_id, comp) in enumerate(zip(step_ids, step_comps)):
                if comp >= 0:
                    # Velocity BC (positive component)
                    vel_ids.append(node_id)
                    vel_comps.append(comp)
                else:
                    # Velocity gradient BC (negative component in legacy format)
                    vgrad_ids.append(node_id)
                    vgrad_comps.append(abs(comp))  # Convert to positive for new format
            # Create velocity BC object
            velocity_bc = None
            if vel_ids:
                velocity_bc = {
                    'essential_ids': vel_ids,
                    'essential_comps': vel_comps,
                    'essential_vals': step_vals
                }
            
            # Create velocity gradient BC object
            vgrad_bc = None
            if vgrad_ids:
                vgrad_bc = {
                    'essential_ids': vgrad_ids,
                    'essential_comps': vgrad_comps,
                    'velocity_gradient': step_vgrads # [item for sublist in step_vgrads for item in sublist]
                }
                # Add origin if not default
                if vgrad_origin != [0.0, 0.0, 0.0]:
                    vgrad_bc['origin'] = vgrad_origin

            return velocity_bc, vgrad_bc

        # Parse essential arrays
        essential_ids = old_bcs.get('essential_ids', [])
        essential_comps = old_bcs.get('essential_comps', [])
        essential_vals = old_bcs.get('essential_vals', [])
        essential_vel_grad = old_bcs.get('essential_vel_grad', [])
        vgrad_origin = old_bcs.get('vgrad_origin', [0.0, 0.0, 0.0])

        # Convert arrays to BC objects
        velocity_bcs = []
        vgrad_bcs = []

        if changing_bcs and self._is_nested_array(essential_ids):
            # Time-dependent case: nested arrays
            for i, step in enumerate(old_bcs['update_steps']):
                step_ids = essential_ids[i] if i < len(essential_ids) else []
                step_comps = essential_comps[i] if i < len(essential_comps) else []
                step_vals = essential_vals[i] if i < len(essential_vals) else []
                step_vgrads = essential_vel_grad[i] if i < len(essential_vel_grad) else []
                
                # Create BC objects for this step
                vel_bc, vgrad_bc = create_bc_objects_for_step(
                    step_ids, step_comps, step_vals, step_vgrads, vgrad_origin
                )
                
                if vel_bc:
                    velocity_bcs.append(vel_bc)
                if vgrad_bc:
                    vgrad_bcs.append(vgrad_bc)
        else:
            # Constant case: flat arrays
            vel_bc, vgrad_bc = create_bc_objects_for_step(
                essential_ids, essential_comps, essential_vals, 
                essential_vel_grad if essential_vel_grad else [], vgrad_origin
            )

            if vel_bc:
                velocity_bcs.append(vel_bc)
            if vgrad_bc:
                vgrad_bcs.append(vgrad_bc)
        # Migrate velocity BCs - handle both old flat format and new structured format
        if velocity_bcs:
            new_bcs['velocity_bcs'] = velocity_bcs
        # Migrate velocity gradient BCs
        if vgrad_bcs:
            new_bcs['velocity_gradient_bcs'] = vgrad_bcs
                
        self.new_config['BCs'] = new_bcs
    
    def _migrate_visualizations(self):
        """Migrate visualization options."""
        if 'Visualizations' not in self.old_config:
            return
            
        old_vis = self.old_config['Visualizations']
        new_vis = {}
        
        # Boolean options with normalization
        for key in ['visit', 'conduit', 'paraview', 'adios2']:
            if key in old_vis:
                new_vis[key] = self._normalize_bool(old_vis[key])
                
        # Direct integer/string mappings
        for key in ['steps', 'floc']:
            if key in old_vis:
                new_vis[key] = old_vis[key]
        
        # Skip fields that are now in PostProcessing
        skip_fields = [
            'avg_stress_fname', 'avg_def_grad_fname', 
            'avg_euler_strain_fname', 'avg_pl_work_fname',
            'additional_avgs', 'light_up', 'light_hkls',
            'light_dist_tol', 'light_s_dir', 'lattice_params',
            'lattice_basename'
        ]
        
        # Copy any other fields not in skip list
        for key, value in old_vis.items():
            if key not in new_vis and key not in skip_fields:
                new_vis[key] = value
                self.warnings.append(f"Unknown visualization field '{key}' copied as-is")
                
        self.new_config['Visualizations'] = new_vis
    
    def _migrate_post_processing(self):
        """Migrate post-processing options (new feature in v2)."""
        post_proc = {
            'volume_averages': {
                'enabled': False,
                'stress': False,
                'def_grad': False,
                'euler_strain': False,
                'plastic_work': False,
                'output_frequency': 1
            },
            'projections': {
                'enabled_projections': [],
                'auto_enable_compatible': True
            }
        }
        
        # Check for average file names in old config Visualizations section
        old_vis = self.old_config.get('Visualizations', {})
        
        # Map old filenames to new boolean flags
        filename_mappings = {
            'avg_stress_fname': 'stress',
            'avg_def_grad_fname': 'def_grad',
            'avg_euler_strain_fname': 'euler_strain',
            'avg_pl_work_fname': 'plastic_work'
        }
        
        # Check if any averaging files are specified
        for old_key, new_key in filename_mappings.items():
            if old_key in old_vis:
                post_proc['volume_averages']['enabled'] = True
                post_proc['volume_averages'][new_key] = True
                # Store the filename for reference
                post_proc['volume_averages'][old_key] = old_vis[old_key]
                
        if self._normalize_bool(old_vis.get('additional_avgs', False)):
            post_proc['volume_averages']['additional_avgs'] = True
            self.warnings.append(
                "additional_avgs detected. Review PostProcessing options for additional averaging capabilities."
            )
            
        # Handle LightUp options
        if self._normalize_bool(old_vis.get('light_up', False)):
            post_proc['light_up'] = {
                'enabled': True
            }
            
            if 'light_hkls' in old_vis:
                post_proc['light_up']['hkl_directions'] = old_vis['light_hkls']
            if 'light_dist_tol' in old_vis:
                post_proc['light_up']['dist_tol'] = old_vis['light_dist_tol']
            if 'light_s_dir' in old_vis:
                post_proc['light_up']['stress_direction'] = old_vis['light_s_dir']
            if 'lattice_params' in old_vis:
                post_proc['light_up']['lattice_params'] = old_vis['lattice_params']
            if 'lattice_basename' in old_vis:
                post_proc['light_up']['basename'] = old_vis['lattice_basename']
        self.new_config['PostProcessing'] = post_proc
    
    def save_new_config(self, filepath: str):
        """Save the migrated configuration with proper formatting."""
        # Create backup of original if saving to same location
        if os.path.exists(filepath):
            backup_path = filepath + '.backup'
            print(f"Creating backup of existing file: {backup_path}")
            os.rename(filepath, backup_path)
            
        # Write with custom formatting to preserve structure
        with open(filepath, 'w') as f:
            f.write(dumps_indented(self.new_config))
            
        print(f"Migrated configuration saved to: {filepath}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

    def generate_modular_files(self, base_dir: str = "."):
        """Generate separate material and post-processing files (optional)."""
        base_path = Path(base_dir)
        
        # Extract materials to separate file
        if 'Materials' in self.new_config and self.new_config['Materials']:
            material_file = base_path / "materials.toml"
            with open(material_file, 'w') as f:
                # Write just the materials
                f.write('# Generated material configuration\n')
                for mat in self.new_config['Materials']:
                    f.write(f'\n[[Materials]]\n')
                    self._write_table_generic(f, mat, 'Materials', indent_level=1, in_materials=True)
            print(f"Generated material file: {material_file}")
            
            # Update main config to reference material file
            self.new_config['materials'] = [str(material_file)]
            del self.new_config['Materials']
            
        # Extract post-processing to separate file
        if 'PostProcessing' in self.new_config:
            pp_file = base_path / "post_processing.toml"
            with open(pp_file, 'w') as f:
                f.write('# Generated post-processing configuration\n')
                f.write('\n[PostProcessing]\n')
                self._write_table_generic(f, self.new_config['PostProcessing'], 
                                        'PostProcessing', indent_level=0, in_materials=False)
            print(f"Generated post-processing file: {pp_file}")
            
            # Update main config to reference post-processing file
            self.new_config['post_processing'] = str(pp_file)
            del self.new_config['PostProcessing']


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ExaConstit option files from old format to new format"
    )
    parser.add_argument(
        "input_file",
        help="Path to old format options.toml file"
    )
    parser.add_argument(
        "-o", "--output",
        default="options_v2.toml",
        help="Output file path (default: options_v2.toml)"
    )
    parser.add_argument(
        "-m", "--modular",
        action="store_true",
        help="Generate separate material and post-processing files"
    )
    parser.add_argument(
        "-d", "--output-dir",
        default=".",
        help="Directory for modular output files (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Perform migration
    migrator = OptionMigrator()
    
    try:
        print(f"Loading old configuration from: {args.input_file}")
        migrator.load_old_config(args.input_file)
        
        print("Migrating configuration...")
        migrator.migrate()
        
        if args.modular:
            print("Generating modular configuration files...")
            migrator.generate_modular_files(args.output_dir)
            
        migrator.save_new_config(args.output)
        
        print("\nMigration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()